"""Backward compatibility stub - import from package."""
from fig_persona_drift.model_server import *  # noqa: F401, F403

if __name__ == "__main__":
    # Support running as script
    import sys
    from fig_persona_drift import model_server
    sys.exit(model_server.main() if hasattr(model_server, 'main') else 0)
