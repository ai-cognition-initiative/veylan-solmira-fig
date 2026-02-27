"""Backward compatibility stub - import from package."""
from fig_persona_drift.generate_conversations import *  # noqa: F401, F403

if __name__ == "__main__":
    import sys
    from fig_persona_drift import generate_conversations
    sys.exit(generate_conversations.main() if hasattr(generate_conversations, 'main') else 0)
