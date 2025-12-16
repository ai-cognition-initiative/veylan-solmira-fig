"""
Basic OpenRouter API test script.
Set your API key as environment variable: export OPENROUTER_API_KEY="your-key-here"
"""

import os
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("Error: Set OPENROUTER_API_KEY environment variable")
    print("  export OPENROUTER_API_KEY='your-key-here'")
    exit(1)

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    },
    json={
        "model": "openai/gpt-3.5-turbo",  # cheap model for testing
        "messages": [
            {"role": "user", "content": "Say hello in exactly 5 words."}
        ],
    }
)

if response.status_code == 200:
    data = response.json()
    message = data["choices"][0]["message"]["content"]
    print(f"Success! Response: {message}")
else:
    print(f"Error {response.status_code}: {response.text}")
