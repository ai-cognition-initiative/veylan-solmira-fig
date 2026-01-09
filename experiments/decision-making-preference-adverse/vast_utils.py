"""
Vast.ai GPU utilities for mechanistic interpretability experiments.

Setup:
    pip install vastai-sdk
    export VASTAI_API_KEY="your-key-here"

    # For Docker image builds:
    docker login

Usage:
    # Interactive
    from vast_utils import VastGPU
    gpu = VastGPU()
    gpu.search_gpus(gpu_name="RTX_4090", max_price=0.50)
    gpu.launch(offer_id=12345, image="veylansolmira/gemma-sae:latest")
    gpu.ssh_command()

    # Automated run (rebuilds image, launches, runs, downloads results, destroys)
    python vast_utils.py run gemma_sae_hello.py
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present (searches up from script location)
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))  # explicitly search parent dirs
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables


# Docker image config
DOCKER_IMAGE = "veylansolmira/gemma-sae:latest"
DOCKERFILE_DIR = Path(__file__).parent

# SSH key for vast.ai instances (set VAST_SSH_KEY in .env)
SSH_KEY_PATH = os.environ.get("VAST_SSH_KEY")


@dataclass
class GPUOffer:
    """Represents an available GPU offer."""
    id: int
    gpu_name: str
    num_gpus: int
    price_per_hour: float
    gpu_ram: float
    reliability: float

    def __str__(self):
        return (f"[{self.id}] {self.num_gpus}x {self.gpu_name} "
                f"({self.gpu_ram:.0f}GB) @ ${self.price_per_hour:.3f}/hr "
                f"(reliability: {self.reliability:.1%})")


class VastGPU:
    """Wrapper for vast.ai SDK focused on mech interp workloads."""

    def __init__(self, api_key: str = None):
        """
        Initialize vast.ai connection.

        Args:
            api_key: Vast.ai API key. If not provided, reads from VASTAI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("VASTAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set VASTAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            from vastai_sdk import VastAI
            self.sdk = VastAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("vastai-sdk not installed. Run: pip install vastai-sdk")

        self.current_instance_id = None

    def search_gpus(
        self,
        gpu_name: str = None,
        min_gpu_ram: float = 24,
        max_price: float = 1.0,
        num_gpus: int = 1,
        min_reliability: float = 0.95,
        limit: int = 10,
    ) -> list[GPUOffer]:
        """
        Search for available GPU instances.

        Args:
            gpu_name: Specific GPU model (e.g., "RTX_4090", "A100", "H100")
            min_gpu_ram: Minimum GPU RAM in GB (default 24 for mech interp)
            max_price: Maximum price per hour in USD
            num_gpus: Number of GPUs required
            min_reliability: Minimum reliability score (0-1)
            limit: Max number of results to return

        Returns:
            List of GPUOffer objects sorted by price
        """
        # Build query string
        query_parts = [
            "rented=False",
            "rentable=True",
            f"num_gpus>={num_gpus}",
            f"gpu_ram>={min_gpu_ram}",
            f"dph_total<={max_price}",
            f"reliability>={min_reliability}",
        ]
        if gpu_name:
            query_parts.append(f"gpu_name={gpu_name}")

        query = " ".join(query_parts)

        try:
            results = self.sdk.search_offers(query=query)
        except Exception as e:
            print(f"Search failed: {e}")
            return []

        # Parse results into GPUOffer objects
        offers = []
        if isinstance(results, list):
            for r in results[:limit]:
                try:
                    # Normalize gpu_name: "RTX 4090" -> "RTX_4090"
                    gpu_name = r.get("gpu_name", "unknown").replace(" ", "_")
                    offers.append(GPUOffer(
                        id=r.get("id"),
                        gpu_name=gpu_name,
                        num_gpus=r.get("num_gpus", 1),
                        price_per_hour=r.get("dph_total", 0),
                        gpu_ram=r.get("gpu_ram", 0),
                        reliability=r.get("reliability", 0),
                    ))
                except Exception:
                    continue

        # Sort by price
        offers.sort(key=lambda x: x.price_per_hour)
        return offers

    def launch(
        self,
        offer_id: int,
        gpu_name: str = "RTX_4090",
        image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        disk_gb: int = 50,
        num_gpus: int = 1,
        env_vars: dict = None,
    ) -> dict:
        """
        Launch a GPU instance.

        Args:
            offer_id: ID from search_gpus results
            gpu_name: GPU model name
            image: Docker image to use
            disk_gb: Disk space in GB
            num_gpus: Number of GPUs
            env_vars: Optional environment variables dict

        Returns:
            Instance creation response
        """
        try:
            # Let SDK find and launch based on gpu_name
            result = self.sdk.launch_instance(
                gpu_name=gpu_name,
                num_gpus=str(num_gpus),
                image=image,
                disk=disk_gb,
                ssh=True,  # Enable SSH access
            )

            # Try to extract instance ID from result
            if isinstance(result, dict) and "new_contract" in result:
                self.current_instance_id = result["new_contract"]

            print(f"Instance launched. ID: {self.current_instance_id}")
            return result

        except Exception as e:
            print(f"Launch failed: {e}")
            return {"error": str(e)}

    def show_instances(self) -> list:
        """List all current instances."""
        try:
            return self.sdk.show_instances()
        except Exception as e:
            print(f"Failed to list instances: {e}")
            return []

    def stop(self, instance_id: int = None):
        """Stop an instance."""
        instance_id = instance_id or self.current_instance_id
        if not instance_id:
            print("No instance ID provided")
            return

        try:
            self.sdk.stop_instance(ID=instance_id)
            print(f"Instance {instance_id} stopped")
        except Exception as e:
            print(f"Failed to stop instance: {e}")

    def destroy(self, instance_id: int = None):
        """Destroy an instance (permanent)."""
        instance_id = instance_id or self.current_instance_id
        if not instance_id:
            print("No instance ID provided")
            return

        try:
            self.sdk.destroy_instance(id=instance_id)
            print(f"Instance {instance_id} destroyed")
            if instance_id == self.current_instance_id:
                self.current_instance_id = None
        except Exception as e:
            print(f"Failed to destroy instance: {e}")

    def ssh_command(self, instance_id: int = None) -> str:
        """Get SSH command for connecting to instance."""
        instances = self.show_instances()
        instance_id = instance_id or self.current_instance_id

        for inst in instances:
            if inst.get("id") == instance_id:
                host = inst.get("ssh_host")
                port = inst.get("ssh_port")
                if host and port:
                    cmd = f"ssh -p {port} root@{host}"
                    print(cmd)
                    return cmd

        print("Could not find SSH info for instance")
        return ""

    def get_ssh_info(self, instance_id: int = None) -> tuple[str, int] | None:
        """Get SSH host and port for an instance."""
        instances = self.show_instances()
        instance_id = instance_id or self.current_instance_id

        for inst in instances:
            if inst.get("id") == instance_id:
                host = inst.get("ssh_host")
                port = inst.get("ssh_port")
                if host and port:
                    return (host, port)
        return None

    def find_running_instance(self) -> dict | None:
        """Find an existing running instance."""
        instances = self.show_instances()
        for inst in instances:
            if inst.get("actual_status") == "running":
                return inst
        return None

    def scp_files(self, files: list[str], instance_id: int = None) -> bool:
        """Copy files to a running instance via SCP."""
        ssh_info = self.get_ssh_info(instance_id)
        if not ssh_info:
            print("Could not get SSH info for instance")
            return False

        host, port = ssh_info

        for file_path in files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            result = subprocess.run(
                ["scp", "-P", str(port), "-i", str(SSH_KEY_PATH),
                 "-o", "StrictHostKeyChecking=no",
                 file_path, f"root@{host}:/app/"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"SCP failed for {file_path}: {result.stderr}")
                return False
            print(f"Copied: {file_path}")

        return True

    def run_remote_command(self, command: str, instance_id: int = None, env_vars: dict = None) -> subprocess.CompletedProcess:
        """Run a command on the remote instance via SSH."""
        ssh_info = self.get_ssh_info(instance_id)
        if not ssh_info:
            print("Could not get SSH info for instance")
            return None

        host, port = ssh_info

        # Prepend environment variables if provided
        if env_vars:
            exports = " && ".join(f"export {k}={v}" for k, v in env_vars.items())
            command = f"{exports} && {command}"

        result = subprocess.run(
            ["ssh", "-p", str(port), "-i", str(SSH_KEY_PATH),
             "-o", "StrictHostKeyChecking=no",
             f"root@{host}", command],
            capture_output=True,
            text=True
        )
        return result


# Recommended GPU configs for different mech interp tasks
# Pricing as of 2026-01-07: RTX 4090 ~$0.24/hr, A100 ~$1.00/hr
RECOMMENDED_CONFIGS = {
    "gemma-2b-sae": {
        "gpu_name": "RTX_4090",
        "min_gpu_ram": 24,
        "max_price": 0.35,
        "notes": "GemmaScope SAEs on Gemma 2B, fits in 24GB"
    },
    "gemma-9b-sae": {
        "gpu_name": "A100",
        "min_gpu_ram": 40,
        "max_price": 1.50,
        "notes": "GemmaScope SAEs on Gemma 9B, needs 40GB+"
    },
    "phi-3-mini": {
        "gpu_name": "RTX_4090",
        "min_gpu_ram": 24,
        "max_price": 0.35,
        "notes": "TransformerLens on Phi-3-mini-4k-instruct"
    },
}


def quick_search(config_name: str = "gemma-2b-sae") -> list[GPUOffer]:
    """Quick search using recommended config."""
    if config_name not in RECOMMENDED_CONFIGS:
        print(f"Unknown config. Available: {list(RECOMMENDED_CONFIGS.keys())}")
        return []

    config = RECOMMENDED_CONFIGS[config_name]
    print(f"Searching for: {config['notes']}")

    gpu = VastGPU()
    return gpu.search_gpus(
        gpu_name=config.get("gpu_name"),
        min_gpu_ram=config["min_gpu_ram"],
        max_price=config["max_price"],
    )


def build_and_push_image(script_name: str = None) -> bool:
    """Build Docker image and push to registry."""
    print("Building Docker image...")

    # Build
    result = subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "."],
        cwd=DOCKERFILE_DIR,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    print("Build complete.")

    # Push
    print("Pushing to Docker Hub...")
    result = subprocess.run(
        ["docker", "push", DOCKER_IMAGE],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Push failed: {result.stderr}")
        return False
    print("Push complete.")

    return True


def run_experiment(
    script_name: str = "gemma_sae_hello.py",
    config: str = "gemma-2b-sae",
    skip_build: bool = False,
    teardown: bool = False,
) -> dict:
    """
    Full automated experiment run:
    1. Check for existing running instance (reuse if found)
    2. If no instance, build/push Docker image and launch new one
    3. SCP script to instance and run
    4. Keep instance running (unless --teardown)

    Args:
        script_name: Python script to run
        config: GPU config from RECOMMENDED_CONFIGS
        skip_build: Skip Docker build/push if launching new instance
        teardown: Destroy instance after run (default: keep running)

    Returns:
        Dict with results or error
    """
    if config not in RECOMMENDED_CONFIGS:
        return {"error": f"Unknown config: {config}"}

    cfg = RECOMMENDED_CONFIGS[config]
    gpu = VastGPU()

    # Step 1: Check for existing running instance
    existing = gpu.find_running_instance()

    if existing:
        instance_id = existing.get("id")
        gpu.current_instance_id = instance_id
        print(f"Found running instance: {instance_id}")
        print(f"  GPU: {existing.get('gpu_name')} @ ${existing.get('dph_total', 0):.3f}/hr")

        # SCP the script and run it
        script_path = DOCKERFILE_DIR / script_name
        if not script_path.exists():
            return {"error": f"Script not found: {script_path}"}

        print(f"\nCopying {script_name} to instance...")
        if not gpu.scp_files([str(script_path)], instance_id):
            return {"error": "Failed to copy script to instance"}

        print(f"Running {script_name}...")
        env_vars = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token
        result = gpu.run_remote_command(f"cd /app && python {script_name}", instance_id, env_vars=env_vars)

        if result:
            print("\n--- Output ---")
            print(result.stdout)
            if result.stderr:
                print("--- Stderr ---")
                print(result.stderr)

    else:
        # No existing instance — launch new one
        print("No running instance found. Launching new one...")

        # Build and push image
        if not skip_build:
            if not build_and_push_image(script_name):
                return {"error": "Docker build/push failed"}

        # Find GPU
        print(f"\nSearching for GPU: {cfg['notes']}")
        offers = gpu.search_gpus(
            gpu_name=cfg.get("gpu_name"),
            min_gpu_ram=cfg["min_gpu_ram"],
            max_price=cfg["max_price"],
        )

        if not offers:
            return {"error": "No GPU offers found"}

        best_offer = offers[0]
        print(f"Selected: {best_offer}")

        # Launch instance
        print("\nLaunching instance...")
        result = gpu.launch(
            offer_id=best_offer.id,
            gpu_name=best_offer.gpu_name,
            num_gpus=best_offer.num_gpus,
            image=DOCKER_IMAGE,
        )

        if "error" in result:
            return result

        instance_id = gpu.current_instance_id
        print(f"Instance {instance_id} launching...")

        # Wait for instance to be ready
        print("Waiting for instance to start...")
        max_wait = 600  # 10 minutes
        poll_interval = 10
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            instances = gpu.show_instances()
            inst = next((i for i in instances if i.get("id") == instance_id), None)

            if not inst:
                print("Instance not found - may have completed or failed")
                break

            status = inst.get("actual_status", "unknown")
            print(f"  Status: {status} ({elapsed}s elapsed)")

            if status == "running":
                print("\nInstance running!")
                break
            elif status in ["exited", "error"]:
                print(f"Instance ended with status: {status}")
                break

    # Teardown if requested
    if teardown:
        print(f"\nTearing down instance {instance_id}...")
        gpu.destroy(instance_id)
        return {"instance_id": instance_id, "status": "destroyed"}

    # Otherwise keep running and print info
    print(f"\n{'='*50}")
    print("INSTANCE RUNNING")
    print(f"{'='*50}")
    print(f"Instance ID: {instance_id}")
    print(f"\nSSH in with:")
    gpu.ssh_command(instance_id)
    print(f"\nDestroy when done:")
    print(f"  python vast_utils.py destroy")

    return {
        "instance_id": instance_id,
        "status": "running",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Vast.ai GPU Utility")
        print("=" * 50)
        print("\nUsage:")
        print("  python vast_utils.py search              # Search for GPUs")
        print("  python vast_utils.py run <script.py>     # Run experiment (reuses existing instance)")
        print("  python vast_utils.py run --skip-build    # Run without rebuilding image (new instance only)")
        print("  python vast_utils.py run --teardown      # Run and destroy instance after")
        print("  python vast_utils.py destroy             # Destroy running instance")
        print("  python vast_utils.py status              # Show running instances")
        print("\nConfigs available:")
        for name, config in RECOMMENDED_CONFIGS.items():
            print(f"  {name}: {config['notes']}")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "search":
        if not os.environ.get("VASTAI_API_KEY"):
            print("Set VASTAI_API_KEY environment variable first.")
            sys.exit(1)
        offers = quick_search("gemma-2b-sae")
        print(f"\nFound {len(offers)} offers:")
        for offer in offers[:5]:
            print(f"  {offer}")

    elif cmd == "run":
        if not os.environ.get("VASTAI_API_KEY"):
            print("Set VASTAI_API_KEY environment variable first.")
            sys.exit(1)

        skip_build = "--skip-build" in sys.argv
        teardown = "--teardown" in sys.argv
        script = next((a for a in sys.argv[2:] if not a.startswith("--")), "gemma_sae_hello.py")

        result = run_experiment(script_name=script, skip_build=skip_build, teardown=teardown)
        print(f"\nResult: {result}")

    elif cmd == "destroy":
        if not os.environ.get("VASTAI_API_KEY"):
            print("Set VASTAI_API_KEY environment variable first.")
            sys.exit(1)

        gpu = VastGPU()
        existing = gpu.find_running_instance()
        if existing:
            instance_id = existing.get("id")
            print(f"Destroying instance {instance_id}...")
            gpu.destroy(instance_id)
        else:
            print("No running instance found.")

    elif cmd == "status":
        if not os.environ.get("VASTAI_API_KEY"):
            print("Set VASTAI_API_KEY environment variable first.")
            sys.exit(1)

        gpu = VastGPU()
        instances = gpu.show_instances()
        if instances:
            print(f"Found {len(instances)} instance(s):")
            for inst in instances:
                print(f"  [{inst.get('id')}] {inst.get('gpu_name')} - {inst.get('actual_status')} @ ${inst.get('dph_total', 0):.3f}/hr")
        else:
            print("No instances found.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
