"""
Vast.ai GPU utilities for the metacognition-persona-drift experiment.

Adapted from the decision-making-preference-adverse experiment's vast_utils.py.

Setup:
    pip install vastai-sdk python-dotenv
    export VASTAI_API_KEY="your-key-here"
    export VAST_SSH_KEY="/path/to/ssh/key"

Usage:
    # Search for GPUs
    python vast_utils.py search --config gemma-27b

    # Run full GPU pipeline (generate + activations)
    python vast_utils.py run --model google/gemma-2-27b-it

    # Test with 3 roles
    python vast_utils.py run --model google/gemma-2-27b-it --roles default assistant detective --question-count 10

    # Launch interactive drift explorer (Gradio chat + activation projection)
    python vast_utils.py explore

    # Download explore session JSONs
    python vast_utils.py download-explore

    # Download pipeline results from running instance
    python vast_utils.py download --model google/gemma-2-27b-it

    # Destroy instance
    python vast_utils.py destroy
"""

import os
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

# Python 3.12+ removed distutils; vastai-sdk still imports it.
# Provide a minimal shim so the SDK can load.
if sys.version_info >= (3, 12):
    try:
        import distutils.util  # noqa: F401
    except ImportError:
        _distutils = types.ModuleType("distutils")
        _distutils_util = types.ModuleType("distutils.util")

        def _strtobool(val):
            val = val.lower()
            if val in ("y", "yes", "t", "true", "on", "1"):
                return 1
            elif val in ("n", "no", "f", "false", "off", "0"):
                return 0
            raise ValueError(f"invalid truth value {val!r}")

        _distutils_util.strtobool = _strtobool
        _distutils.util = _distutils_util
        sys.modules["distutils"] = _distutils
        sys.modules["distutils.util"] = _distutils_util

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

# Docker images
BASE_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
CUSTOM_IMAGE = "veylansolmira/assistant-axis:latest"  # for future production use
EXPERIMENT_DIR = Path(__file__).parent

# SSH key for vast.ai instances
SSH_KEY_PATH = os.environ.get("VAST_SSH_KEY")


def _read_ssh_public_key() -> str | None:
    """Read SSH public key from the configured path."""
    if not SSH_KEY_PATH:
        return None
    pub_key_path = SSH_KEY_PATH + ".pub"
    if os.path.exists(pub_key_path):
        with open(pub_key_path, 'r') as f:
            return f.read().strip()
    return None


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
        ram_gb = self.gpu_ram / 1024  # API reports MB
        return (f"[{self.id}] {self.num_gpus}x {self.gpu_name} "
                f"({ram_gb:.0f}GB) @ ${self.price_per_hour:.3f}/hr "
                f"(reliability: {self.reliability:.1%})")


class VastGPU:
    """Wrapper for vast.ai SDK focused on assistant axis workloads."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("VASTAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key. Set VASTAI_API_KEY environment variable "
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
        """Search for available GPU instances.

        Note: gpu_ram filter is applied client-side because the vast.ai
        search API uses inconsistent units for gpu_ram in queries vs responses.
        """
        query_parts = [
            "rented=False",
            "rentable=True",
            f"num_gpus>={num_gpus}",
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

        offers = []
        if isinstance(results, list):
            for r in results:
                # Client-side gpu_ram filter (response reports MB)
                if r.get("gpu_ram", 0) < min_gpu_ram:
                    continue
                try:
                    gpu_name_str = r.get("gpu_name", "unknown").replace(" ", "_")
                    offers.append(GPUOffer(
                        id=r.get("id"),
                        gpu_name=gpu_name_str,
                        num_gpus=r.get("num_gpus", 1),
                        price_per_hour=r.get("dph_total", 0),
                        gpu_ram=r.get("gpu_ram", 0),
                        reliability=r.get("reliability", 0),
                    ))
                except Exception:
                    continue

        offers.sort(key=lambda x: x.price_per_hour)
        return offers[:limit]

    def launch(
        self,
        offer_id: int,
        image: str = BASE_IMAGE,
        disk_gb: int = 250,
        **kwargs,
    ) -> dict:
        """Launch a GPU instance from a specific offer ID."""
        try:
            result = self.sdk.create_instance(
                id=offer_id,
                image=image,
                disk=disk_gb,
                ssh=True,
            )

            if isinstance(result, dict) and "new_contract" in result:
                self.current_instance_id = result["new_contract"]

            print(f"Instance launched. ID: {self.current_instance_id}")

            ssh_pub_key = _read_ssh_public_key()
            if ssh_pub_key and self.current_instance_id:
                print("Attaching SSH key...")
                try:
                    self.sdk.attach_ssh(instance_id=self.current_instance_id, ssh_key=ssh_pub_key)
                    print("SSH key attached.")
                except Exception as e:
                    print(f"Warning: Failed to attach SSH key: {e}")

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

    def ensure_ssh_key_attached(self, instance_id: int = None) -> bool:
        """Ensure SSH key is attached to instance."""
        instance_id = instance_id or self.current_instance_id
        if not instance_id:
            return False

        ssh_pub_key = _read_ssh_public_key()
        if not ssh_pub_key:
            print("Warning: No SSH public key found")
            return False

        try:
            self.sdk.attach_ssh(instance_id=instance_id, ssh_key=ssh_pub_key)
            return True
        except Exception:
            return True  # May fail if already attached

    def scp_to(self, files: list[str], instance_id: int = None, remote_dir: str = "/app/", retries: int = 3) -> bool:
        """Copy files to a running instance via SCP."""
        ssh_info = self.get_ssh_info(instance_id)
        if not ssh_info:
            print("Could not get SSH info")
            return False

        host, port = ssh_info

        for file_path in files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            for attempt in range(retries):
                result = subprocess.run(
                    ["scp", "-P", str(port), "-i", str(SSH_KEY_PATH),
                     "-o", "StrictHostKeyChecking=no",
                     file_path, f"root@{host}:{remote_dir}"],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    print(f"Copied: {file_path}")
                    break
                if "Permission denied" in result.stderr or "Connection closed" in result.stderr:
                    if attempt < retries - 1:
                        print(f"SCP retry ({attempt + 1}/{retries})...")
                        time.sleep(3)
                        continue
                print(f"SCP failed for {file_path}: {result.stderr}")
                return False

        return True

    def scp_from(self, remote_path: str, local_path: str, instance_id: int = None, retries: int = 3) -> bool:
        """Download files/dirs from a running instance via SCP."""
        ssh_info = self.get_ssh_info(instance_id)
        if not ssh_info:
            print("Could not get SSH info")
            return False

        host, port = ssh_info
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

        for attempt in range(retries):
            result = subprocess.run(
                ["scp", "-r", "-P", str(port), "-i", str(SSH_KEY_PATH),
                 "-o", "StrictHostKeyChecking=no",
                 f"root@{host}:{remote_path}", local_path],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                print(f"Downloaded: {remote_path} -> {local_path}")
                return True
            if "Permission denied" in result.stderr or "Connection closed" in result.stderr:
                if attempt < retries - 1:
                    print(f"SCP retry ({attempt + 1}/{retries})...")
                    time.sleep(3)
                    continue
            print(f"SCP failed: {result.stderr}")
            return False
        return False

    def run_remote(self, command: str, instance_id: int = None, env_vars: dict = None, retries: int = 3) -> subprocess.CompletedProcess:
        """Run a command on the remote instance via SSH."""
        ssh_info = self.get_ssh_info(instance_id)
        if not ssh_info:
            print("Could not get SSH info")
            return None

        host, port = ssh_info

        if env_vars:
            exports = " && ".join(f"export {k}={v}" for k, v in env_vars.items())
            command = f"{exports} && {command}"

        for attempt in range(retries):
            result = subprocess.run(
                ["ssh", "-p", str(port), "-i", str(SSH_KEY_PATH),
                 "-o", "StrictHostKeyChecking=no",
                 f"root@{host}", command],
                capture_output=True, text=True,
            )
            if result.returncode == 255 and "Permission denied" in result.stderr:
                if attempt < retries - 1:
                    print(f"SSH retry ({attempt + 1}/{retries})...")
                    time.sleep(3)
                    continue
            return result
        return result

    def ssh_command(self, instance_id: int = None) -> str:
        """Get SSH command for connecting to instance."""
        ssh_info = self.get_ssh_info(instance_id)
        if not ssh_info:
            print("Could not find SSH info")
            return ""
        host, port = ssh_info
        cmd = f"ssh -p {port} -i {SSH_KEY_PATH} root@{host}"
        print(cmd)
        return cmd

    def wait_for_ready(self, instance_id: int = None, max_wait: int = 600, poll_interval: int = 10) -> bool:
        """Wait for instance to become running."""
        instance_id = instance_id or self.current_instance_id
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            instances = self.show_instances()
            inst = next((i for i in instances if i.get("id") == instance_id), None)

            if not inst:
                print("Instance not found")
                return False

            status = inst.get("actual_status", "unknown")
            print(f"  Status: {status} ({elapsed}s)")

            if status == "running":
                print("Instance running!")
                return True
            elif status in ("exited", "error"):
                print(f"Instance failed: {status}")
                return False

        print(f"Timeout after {max_wait}s")
        return False


# GPU configurations for assistant axis models
# Gemma 27B: ~54GB in bf16, needs 80GB GPU
# Qwen 32B: ~64GB in bf16, needs 80GB GPU
# Llama 70B: ~140GB in bf16, needs 2x80GB GPU
# Note: vast.ai gpu_ram is in MB. A100 40GB = 40960, A100 80GB = 81920.
RECOMMENDED_CONFIGS = {
    "gemma-27b": {
        "min_gpu_ram": 81000,
        "max_price": 1.00,
        "num_gpus": 1,
        "model": "google/gemma-2-27b-it",
        "notes": "Gemma 2 27B — 1x 80GB+ GPU",
    },
    "qwen-32b": {
        "min_gpu_ram": 81000,
        "max_price": 1.00,
        "num_gpus": 1,
        "model": "Qwen/Qwen3-32B",
        "notes": "Qwen 3 32B — 1x 80GB+ GPU",
    },
    "llama-70b": {
        "min_gpu_ram": 81000,
        "max_price": 4.00,
        "num_gpus": 2,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "notes": "Llama 3.3 70B — 2x 80GB+ GPU with tensor parallelism",
    },
}


def quick_search(config_name: str = "gemma-27b") -> list[GPUOffer]:
    """Quick search using recommended config."""
    if config_name not in RECOMMENDED_CONFIGS:
        print(f"Unknown config. Available: {list(RECOMMENDED_CONFIGS.keys())}")
        return []

    cfg = RECOMMENDED_CONFIGS[config_name]
    print(f"Searching for: {cfg['notes']}")

    gpu = VastGPU()
    return gpu.search_gpus(
        gpu_name=cfg.get("gpu_name"),
        min_gpu_ram=cfg["min_gpu_ram"],
        max_price=cfg["max_price"],
        num_gpus=cfg.get("num_gpus", 1),
    )


def setup_instance(gpu: "VastGPU", instance_id: int) -> bool:
    """Set up a base image instance with our code and dependencies.

    This is the iterative approach: launch with a known-good base image,
    SCP code, install deps. Faster feedback than building a custom image
    locally on ARM and pushing. Once proven, update the Dockerfile to match.
    """
    print("\n--- Setting up instance ---")

    # 1. Create working directory + install Python deps
    #    Combined into one command so mkdir benefits from SSH retry logic
    print("Installing dependencies...")
    install_cmd = (
        "mkdir -p /app/outputs && "
        "pip install --no-cache-dir "
        "vllm transformers>=4.40 accelerate huggingface_hub "
        "scikit-learn numpy jsonlines tqdm pyarrow"
    )
    result = gpu.run_remote(install_cmd, instance_id)
    if result and result.returncode != 0:
        print(f"Dep install failed:\n{result.stderr[-2000:]}")
        return False
    print("Dependencies installed.")

    # 2. SCP assistant-axis repo (data + package)
    #    Use tar to preserve directory structure and skip .git
    print("Packaging assistant-axis for upload...")
    tar_path = EXPERIMENT_DIR / ".tmp-assistant-axis.tar.gz"
    aa_dir = EXPERIMENT_DIR / "assistant-axis"
    tar_result = subprocess.run(
        ["tar", "czf", str(tar_path),
         "--exclude=.git", "--exclude=notebooks", "--exclude=transcripts",
         "--exclude=__pycache__", "--exclude=.venv",
         "-C", str(EXPERIMENT_DIR), "assistant-axis"],
        capture_output=True, text=True,
    )
    if tar_result.returncode != 0:
        print(f"tar failed: {tar_result.stderr}")
        return False

    tar_size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"Uploading assistant-axis ({tar_size_mb:.1f} MB)...")
    if not gpu.scp_to([str(tar_path)], instance_id, remote_dir="/app/"):
        return False
    tar_path.unlink()  # clean up local tar

    # Extract on remote and install as editable package
    result = gpu.run_remote(
        "cd /app && tar xzf .tmp-assistant-axis.tar.gz && rm .tmp-assistant-axis.tar.gz "
        "&& cd assistant-axis && pip install --no-cache-dir -e .",
        instance_id,
    )
    if result and result.returncode != 0:
        print(f"Remote extract/install failed:\n{result.stderr[-2000:]}")
        return False
    print("assistant-axis installed.")

    # 3. SCP gpu_pipeline.py
    print("Uploading gpu_pipeline.py...")
    if not gpu.scp_to([str(EXPERIMENT_DIR / "gpu_pipeline.py")], instance_id, remote_dir="/app/"):
        return False

    # 4. Verify
    result = gpu.run_remote(
        "python -c \"from assistant_axis import axis; print('OK')\" "
        "&& ls /app/assistant-axis/data/extraction_questions.jsonl "
        "&& ls /app/gpu_pipeline.py",
        instance_id,
    )
    if result and result.returncode == 0:
        print("Setup verified.\n")
        return True
    else:
        print(f"Verification failed:\n{result.stderr if result else 'no result'}")
        return False


def setup_explore_instance(gpu: "VastGPU", instance_id: int) -> bool:
    """Set up instance for the interactive drift explorer.

    Like setup_instance() but installs gradio instead of vllm,
    and uploads explore.py + precomputed axis instead of gpu_pipeline.py.
    """
    print("\n--- Setting up explore instance ---")

    # 1. Install deps (gradio instead of vllm)
    print("Installing dependencies...")
    install_cmd = (
        "mkdir -p /app/explore-sessions && "
        "pip install --no-cache-dir "
        "gradio matplotlib transformers>=4.40 accelerate huggingface_hub "
        "scikit-learn numpy jsonlines tqdm pyarrow"
    )
    result = gpu.run_remote(install_cmd, instance_id)
    if result and result.returncode != 0:
        print(f"Dep install failed:\n{result.stderr[-2000:]}")
        return False
    print("Dependencies installed.")

    # 2. SCP assistant-axis repo (same as setup_instance)
    print("Packaging assistant-axis for upload...")
    tar_path = EXPERIMENT_DIR / ".tmp-assistant-axis.tar.gz"
    tar_result = subprocess.run(
        ["tar", "czf", str(tar_path),
         "--exclude=.git", "--exclude=notebooks", "--exclude=transcripts",
         "--exclude=__pycache__", "--exclude=.venv",
         "-C", str(EXPERIMENT_DIR), "assistant-axis"],
        capture_output=True, text=True,
    )
    if tar_result.returncode != 0:
        print(f"tar failed: {tar_result.stderr}")
        return False

    tar_size_mb = tar_path.stat().st_size / (1024 * 1024)
    print(f"Uploading assistant-axis ({tar_size_mb:.1f} MB)...")
    if not gpu.scp_to([str(tar_path)], instance_id, remote_dir="/app/"):
        return False
    tar_path.unlink()

    result = gpu.run_remote(
        "cd /app && tar xzf .tmp-assistant-axis.tar.gz && rm .tmp-assistant-axis.tar.gz "
        "&& cd assistant-axis && pip install --no-cache-dir -e .",
        instance_id,
    )
    if result and result.returncode != 0:
        print(f"Remote extract/install failed:\n{result.stderr[-2000:]}")
        return False
    print("assistant-axis installed.")

    # 3. SCP explore.py and precomputed axis
    print("Uploading explore.py and axis...")
    files = [
        str(EXPERIMENT_DIR / "explore.py"),
        str(EXPERIMENT_DIR / "data" / "precomputed-axes" / "gemma-2-27b.pt"),
    ]
    if not gpu.scp_to(files, instance_id, remote_dir="/app/"):
        return False

    # 4. Verify
    result = gpu.run_remote(
        "python -c \"from assistant_axis import load_axis; print('OK')\" "
        "&& ls /app/explore.py /app/gemma-2-27b.pt",
        instance_id,
    )
    if result and result.returncode == 0:
        print("Setup verified.\n")
        return True
    else:
        print(f"Verification failed:\n{result.stderr if result else 'no result'}")
        return False


def run_gpu_pipeline(
    model: str = "google/gemma-2-27b-it",
    config: str = "gemma-27b",
    roles: list[str] = None,
    question_count: int = 240,
    step: str = "both",
    teardown: bool = False,
) -> dict:
    """
    Full automated run using base image + SCP approach:
    1. Check for existing instance (reuse if found, skip setup)
    2. If no instance, launch with base PyTorch image
    3. SCP code and install deps on the instance
    4. Run GPU pipeline remotely
    5. Keep instance running (unless teardown=True)
    """
    if config not in RECOMMENDED_CONFIGS:
        return {"error": f"Unknown config: {config}"}

    cfg = RECOMMENDED_CONFIGS[config]
    gpu = VastGPU()
    needs_setup = False

    # Check for existing instance
    existing = gpu.find_running_instance()

    if existing:
        instance_id = existing.get("id")
        gpu.current_instance_id = instance_id
        print(f"Found running instance: {instance_id}")
        print(f"  GPU: {existing.get('gpu_name')} @ ${existing.get('dph_total', 0):.3f}/hr")
        gpu.ensure_ssh_key_attached(instance_id)

        # Check if already set up
        result = gpu.run_remote("test -f /app/gpu_pipeline.py && echo 'ready'", instance_id)
        if result and "ready" in (result.stdout or ""):
            print("Instance already set up.")
        else:
            print("Instance running but not set up.")
            needs_setup = True
    else:
        print("No running instance. Launching with base image...")

        offers = gpu.search_gpus(
            gpu_name=cfg.get("gpu_name"),
            min_gpu_ram=cfg["min_gpu_ram"],
            max_price=cfg["max_price"],
            num_gpus=cfg.get("num_gpus", 1),
        )
        if not offers:
            return {"error": "No GPU offers found"}

        best = offers[0]
        print(f"Selected: {best}")

        result = gpu.launch(
            offer_id=best.id,
            gpu_name=best.gpu_name,
            num_gpus=best.num_gpus,
            image=BASE_IMAGE,
        )
        if "error" in result:
            return result

        instance_id = gpu.current_instance_id
        print(f"Waiting for instance {instance_id}...")

        if not gpu.wait_for_ready(instance_id):
            return {"error": "Instance failed to start"}

        needs_setup = True

    # Setup if needed
    if needs_setup:
        if not setup_instance(gpu, instance_id):
            return {"error": "Instance setup failed"}

    # Build remote command
    cmd_parts = [
        "cd /app && python gpu_pipeline.py",
        f"--model {model}",
        f"--question_count {question_count}",
        f"--step {step}",
    ]
    if roles:
        cmd_parts.append(f"--roles {' '.join(roles)}")

    remote_cmd = " ".join(cmd_parts)

    # Pass tokens for model download (HF) and judge scoring (OpenAI-compatible)
    env_vars = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    # Judge uses OpenAI SDK. Route through OpenRouter if no direct OpenAI key.
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        env_vars["OPENAI_API_KEY"] = openai_key
    else:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            env_vars["OPENAI_API_KEY"] = openrouter_key
            env_vars["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    print(f"\nRunning: {remote_cmd}")
    result = gpu.run_remote(remote_cmd, instance_id, env_vars=env_vars)

    if result:
        print("\n--- Output ---")
        print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
        if result.stderr:
            print("--- Stderr (last 2000 chars) ---")
            print(result.stderr[-2000:])

    if teardown:
        print(f"\nDestroying instance {instance_id}...")
        gpu.destroy(instance_id)
        return {"instance_id": instance_id, "status": "destroyed"}

    print(f"\nInstance {instance_id} still running.")
    gpu.ssh_command(instance_id)
    return {"instance_id": instance_id, "status": "running"}


def download_results(
    model: str = "google/gemma-2-27b-it",
    local_output_dir: Path = None,
) -> bool:
    """Download GPU pipeline results from running instance."""
    gpu = VastGPU()
    existing = gpu.find_running_instance()
    if not existing:
        print("No running instance found.")
        return False

    instance_id = existing.get("id")
    gpu.current_instance_id = instance_id
    gpu.ensure_ssh_key_attached(instance_id)

    model_short = model.split("/")[-1].lower()
    remote_path = f"/app/outputs/{model_short}"

    if local_output_dir is None:
        local_output_dir = EXPERIMENT_DIR / "outputs"
    local_path = str(local_output_dir / model_short)

    print(f"Downloading {remote_path} -> {local_path}")
    return gpu.scp_from(remote_path, local_path, instance_id)


def run_explore(config: str = "gemma-27b") -> dict:
    """Launch the interactive drift explorer on a GPU instance.

    Finds or creates an instance, sets up explore dependencies,
    launches explore.py via nohup, and polls for the Gradio share URL.
    """
    if config not in RECOMMENDED_CONFIGS:
        return {"error": f"Unknown config: {config}"}

    cfg = RECOMMENDED_CONFIGS[config]
    gpu = VastGPU()
    needs_setup = False

    # Check for existing instance
    existing = gpu.find_running_instance()

    if existing:
        instance_id = existing.get("id")
        gpu.current_instance_id = instance_id
        print(f"Found running instance: {instance_id}")
        print(f"  GPU: {existing.get('gpu_name')} @ ${existing.get('dph_total', 0):.3f}/hr")
        gpu.ensure_ssh_key_attached(instance_id)

        result = gpu.run_remote("test -f /app/explore.py && echo 'ready'", instance_id)
        if result and "ready" in (result.stdout or ""):
            print("Instance already set up for explore.")
        else:
            needs_setup = True
    else:
        print("No running instance. Launching with base image...")

        offers = gpu.search_gpus(
            gpu_name=cfg.get("gpu_name"),
            min_gpu_ram=cfg["min_gpu_ram"],
            max_price=cfg["max_price"],
            num_gpus=cfg.get("num_gpus", 1),
        )
        if not offers:
            return {"error": "No GPU offers found"}

        best = offers[0]
        print(f"Selected: {best}")

        result = gpu.launch(
            offer_id=best.id,
            gpu_name=best.gpu_name,
            num_gpus=best.num_gpus,
            image=BASE_IMAGE,
        )
        if "error" in result:
            return result

        instance_id = gpu.current_instance_id
        print(f"Waiting for instance {instance_id}...")

        if not gpu.wait_for_ready(instance_id):
            return {"error": "Instance failed to start"}

        needs_setup = True

    if needs_setup:
        if not setup_explore_instance(gpu, instance_id):
            return {"error": "Instance setup failed"}

    # Launch explore.py via nohup (long-lived server)
    env_vars = {
        # Use /dev/shm for HF cache — the container root overlay is too small
        # for Gemma 27B weights (~54 GB). /dev/shm is 125 GB on A100 instances.
        "HF_HOME": "/dev/shm/huggingface",
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    launch_cmd = (
        "nohup python /app/explore.py "
        f"--model {cfg['model']} "
        "--axis /app/gemma-2-27b.pt "
        "> /app/explore.log 2>&1 &"
    )

    print("Launching explore.py...")
    gpu.run_remote(launch_cmd, instance_id, env_vars=env_vars)

    # Poll for Gradio share URL
    print("Waiting for Gradio to start (model loading may take ~2 min)...")
    for i in range(60):  # up to 5 min
        time.sleep(5)
        result = gpu.run_remote("tail -20 /app/explore.log 2>/dev/null", instance_id)
        if result and result.stdout:
            for line in result.stdout.splitlines():
                if "gradio.live" in line or "Running on" in line:
                    print(line.strip())
                if "gradio.live" in line:
                    print(f"\n{'=' * 60}")
                    print("Explorer is ready! Open the URL above in your browser.")
                    print(f"{'=' * 60}")
                    return {"instance_id": instance_id, "status": "running"}
            if "Error" in result.stdout or "Traceback" in result.stdout:
                print(f"Error detected:\n{result.stdout[-2000:]}")
                return {"error": "explore.py crashed", "log": result.stdout}
        print(f"  Still loading... ({(i + 1) * 5}s)")

    print("Timeout waiting for Gradio. Check logs:")
    gpu.ssh_command(instance_id)
    return {"instance_id": instance_id, "status": "timeout"}


def download_explore_sessions(local_dir: Path = None) -> bool:
    """Download explore session JSONs from running instance."""
    gpu = VastGPU()
    existing = gpu.find_running_instance()
    if not existing:
        print("No running instance found.")
        return False

    instance_id = existing.get("id")
    gpu.current_instance_id = instance_id
    gpu.ensure_ssh_key_attached(instance_id)

    if local_dir is None:
        local_dir = EXPERIMENT_DIR / "data" / "explore-sessions"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading sessions to {local_dir}")
    return gpu.scp_from("/app/explore-sessions/", str(local_dir), instance_id)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Assistant Axis GPU Utility")
        print("=" * 50)
        print("\nUsage:")
        print("  python vast_utils.py search [--config gemma-27b]")
        print("  python vast_utils.py run [--roles default assistant] [--question-count 10]")
        print("  python vast_utils.py setup              # just launch + install deps, no pipeline run")
        print("  python vast_utils.py explore             # launch interactive drift explorer")
        print("  python vast_utils.py download-explore     # download explore session JSONs")
        print("  python vast_utils.py ssh                # print SSH command for running instance")
        print("  python vast_utils.py download [--model google/gemma-2-27b-it]")
        print("  python vast_utils.py destroy")
        print("  python vast_utils.py status")
        print("\nConfigs:")
        for name, cfg in RECOMMENDED_CONFIGS.items():
            print(f"  {name}: {cfg['notes']} (max ${cfg['max_price']:.2f}/hr)")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "search":
        config = "gemma-27b"
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--config" and i + 1 < len(sys.argv) - 2:
                config = sys.argv[i + 3]
        offers = quick_search(config)
        print(f"\nFound {len(offers)} offers:")
        for offer in offers[:5]:
            print(f"  {offer}")

    elif cmd == "run":
        teardown = "--teardown" in sys.argv

        model = "google/gemma-2-27b-it"
        roles = None
        question_count = 240
        config = "gemma-27b"
        step = "both"

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--model" and i + 1 < len(args):
                model = args[i + 1]
                i += 2
            elif args[i] == "--config" and i + 1 < len(args):
                config = args[i + 1]
                i += 2
            elif args[i] == "--roles":
                roles = []
                i += 1
                while i < len(args) and not args[i].startswith("--"):
                    roles.append(args[i])
                    i += 1
            elif args[i] == "--question-count" and i + 1 < len(args):
                question_count = int(args[i + 1])
                i += 2
            elif args[i] == "--step" and i + 1 < len(args):
                step = args[i + 1]
                i += 2
            else:
                i += 1

        result = run_gpu_pipeline(
            model=model,
            config=config,
            roles=roles,
            question_count=question_count,
            step=step,
            teardown=teardown,
        )
        print(f"\nResult: {result}")

    elif cmd == "setup":
        # Launch + setup only (no pipeline run). Good for interactive debugging.
        config = "gemma-27b"
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--config" and i + 1 < len(sys.argv) - 2:
                config = sys.argv[i + 3]

        cfg = RECOMMENDED_CONFIGS[config]
        gpu = VastGPU()
        existing = gpu.find_running_instance()

        if existing:
            instance_id = existing.get("id")
            gpu.current_instance_id = instance_id
            print(f"Found running instance: {instance_id}")
            gpu.ensure_ssh_key_attached(instance_id)
        else:
            print(f"Launching {config} instance...")
            offers = gpu.search_gpus(
                gpu_name=cfg.get("gpu_name"),
                min_gpu_ram=cfg["min_gpu_ram"],
                max_price=cfg["max_price"],
                num_gpus=cfg.get("num_gpus", 1),
            )
            if not offers:
                print("No offers found.")
                sys.exit(1)
            best = offers[0]
            print(f"Selected: {best}")
            gpu.launch(offer_id=best.id, gpu_name=best.gpu_name,
                       num_gpus=best.num_gpus, image=BASE_IMAGE)
            instance_id = gpu.current_instance_id
            if not gpu.wait_for_ready(instance_id):
                sys.exit(1)

        setup_instance(gpu, instance_id)
        gpu.ssh_command(instance_id)

    elif cmd == "ssh":
        gpu = VastGPU()
        existing = gpu.find_running_instance()
        if existing:
            gpu.current_instance_id = existing.get("id")
            gpu.ssh_command(existing.get("id"))
        else:
            print("No running instance found.")

    elif cmd == "download":
        model = "google/gemma-2-27b-it"
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--model" and i + 1 < len(sys.argv) - 2:
                model = sys.argv[i + 3]
        download_results(model=model)

    elif cmd == "explore":
        config = "gemma-27b"
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--config" and i + 1 < len(sys.argv) - 2:
                config = sys.argv[i + 3]
        result = run_explore(config=config)
        print(f"\nResult: {result}")

    elif cmd == "download-explore":
        download_explore_sessions()

    elif cmd == "destroy":
        gpu = VastGPU()
        existing = gpu.find_running_instance()
        if existing:
            instance_id = existing.get("id")
            print(f"Destroying instance {instance_id}...")
            gpu.destroy(instance_id)
        else:
            print("No running instance found.")

    elif cmd == "status":
        gpu = VastGPU()
        instances = gpu.show_instances()
        if instances:
            print(f"Found {len(instances)} instance(s):")
            for inst in instances:
                print(f"  [{inst.get('id')}] {inst.get('gpu_name')} - "
                      f"{inst.get('actual_status')} @ ${inst.get('dph_total', 0):.3f}/hr")
        else:
            print("No instances found.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
