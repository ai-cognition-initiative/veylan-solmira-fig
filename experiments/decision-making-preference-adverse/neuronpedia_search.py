"""
Search Neuronpedia for SAE features matching keywords.

Searches GemmaScope features by keyword to identify candidates for
differential activation analysis (baseline vs adversarial prompts).

Setup:
    1. Get API key from https://neuronpedia.org/account
    2. export NEURONPEDIA_API_KEY="your-key-here"

Usage:
    python neuronpedia_search.py                    # Search all keywords
    python neuronpedia_search.py --keyword prefer   # Search single keyword
    python neuronpedia_search.py --list-models      # List available models
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests


NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org/api"

# Default model to search (Gemma 2 2B with GemmaScope)
DEFAULT_MODEL = "gemma-2-2b"


@dataclass
class Feature:
    """Represents a Neuronpedia SAE feature."""
    model_id: str
    layer: str
    index: int
    description: str
    keyword: str  # The keyword that found this feature

    def __str__(self):
        return f"[{self.model_id}/{self.layer}#{self.index}] {self.description}"

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "layer": self.layer,
            "index": self.index,
            "description": self.description,
            "keyword": self.keyword,
        }


class NeuronpediaSearch:
    """Search Neuronpedia for SAE features."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
        if not self.api_key:
            print("Warning: No API key provided. Some endpoints may be rate-limited.")
            print("Get your key at https://neuronpedia.org/account")

        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search_model(self, model_id: str, query: str, limit: int = 10) -> list[Feature]:
        """
        Search for features by keyword within a specific model.

        Args:
            model_id: Model identifier (e.g., "gemma-2-2b")
            query: Search query (min 3 characters)
            limit: Max results to return

        Returns:
            List of Feature objects
        """
        if len(query) < 3:
            print(f"Query too short (min 3 chars): {query}")
            return []

        url = f"{NEURONPEDIA_BASE_URL}/explanation/search-model"
        payload = {
            "modelId": model_id,
            "query": query,
        }

        try:
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"API error for query '{query}': {e}")
            return []

        features = []
        results = data if isinstance(data, list) else data.get("results", [])

        for item in results[:limit]:
            try:
                features.append(Feature(
                    model_id=model_id,
                    layer=item.get("layer", item.get("source", "unknown")),
                    index=item.get("index", 0),
                    description=item.get("description", item.get("explanation", "")),
                    keyword=query,
                ))
            except Exception:
                continue

        return features

    def search_keywords(
        self,
        keywords: list[str],
        model_id: str = DEFAULT_MODEL,
        limit_per_keyword: int = 5,
        delay: float = 0.5,
    ) -> dict[str, list[Feature]]:
        """
        Search multiple keywords and collect features.

        Args:
            keywords: List of keywords to search
            model_id: Model to search
            limit_per_keyword: Max features per keyword
            delay: Seconds between requests (rate limiting)

        Returns:
            Dict mapping keyword -> list of features
        """
        results = {}

        for keyword in keywords:
            print(f"Searching: {keyword}...")
            features = self.search_model(model_id, keyword, limit=limit_per_keyword)
            results[keyword] = features
            print(f"  Found {len(features)} features")

            if delay > 0:
                time.sleep(delay)

        return results


def load_keywords(keywords_file: Path = None) -> dict:
    """Load keywords from JSON file."""
    if keywords_file is None:
        keywords_file = Path(__file__).parent / "data/neuronpedia_keywords.json"

    with open(keywords_file) as f:
        return json.load(f)


def flatten_keywords(keywords_data: dict) -> list[str]:
    """Flatten categorized keywords into single list."""
    all_keywords = []
    for category, keywords in keywords_data.get("categories", {}).items():
        all_keywords.extend(keywords)
    return list(set(all_keywords))  # Dedupe


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Search Neuronpedia for SAE features")
    parser.add_argument("--keyword", "-k", type=str, help="Search single keyword")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                        help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", "-n", type=int, default=5,
                        help="Max results per keyword (default: 5)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--list-models", action="store_true",
                        help="List commonly available models")

    args = parser.parse_args()

    if args.list_models:
        print("Common GemmaScope models:")
        print("  gemma-2-2b       - Gemma 2 2B (recommended starting point)")
        print("  gemma-2-9b       - Gemma 2 9B")
        print("  gemma-2-27b      - Gemma 2 27B")
        print("\nSee https://neuronpedia.org/available-resources for full list")
        return

    searcher = NeuronpediaSearch()

    if args.keyword:
        # Single keyword search
        features = searcher.search_model(args.model, args.keyword, limit=args.limit)
        print(f"\nFound {len(features)} features for '{args.keyword}':")
        for f in features:
            print(f"  {f}")
    else:
        # Search all keywords from file
        keywords_data = load_keywords()
        keywords = flatten_keywords(keywords_data)

        print(f"Searching {len(keywords)} keywords in {args.model}...")
        print("=" * 50)

        results = searcher.search_keywords(
            keywords,
            model_id=args.model,
            limit_per_keyword=args.limit,
        )

        # Summary
        total = sum(len(f) for f in results.values())
        print("=" * 50)
        print(f"Total features found: {total}")

        # Dedupe by (layer, index)
        unique_features = {}
        for keyword, features in results.items():
            for f in features:
                key = (f.layer, f.index)
                if key not in unique_features:
                    unique_features[key] = f

        print(f"Unique features: {len(unique_features)}")

        # Output
        if args.output:
            output_data = {
                "model": args.model,
                "keywords_searched": keywords,
                "features": [f.to_dict() for f in unique_features.values()],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to {args.output}")
        else:
            print("\nTop features:")
            for f in list(unique_features.values())[:20]:
                print(f"  {f}")


if __name__ == "__main__":
    main()
