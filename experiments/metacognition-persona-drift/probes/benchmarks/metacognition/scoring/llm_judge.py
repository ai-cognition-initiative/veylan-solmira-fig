"""
LLM-as-judge scoring for open-ended metacognition benchmark items.

Uses a judge model to evaluate phenomenological and other open-ended responses
according to predefined rubrics.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os


@dataclass
class JudgmentResult:
    """Result from LLM judge evaluation."""
    item_id: str
    scores: Dict[str, int]  # dimension_name -> score (1-5)
    justifications: Dict[str, str]  # dimension_name -> explanation
    total_score: float
    raw_response: str
    confidence: float  # judge's confidence in its scoring


def get_rubric_dimensions(subdomain: str) -> Dict[str, Dict]:
    """
    Get the relevant rubric dimensions for a subdomain.

    Args:
        subdomain: The subdomain being evaluated

    Returns:
        Dictionary of dimension definitions
    """
    rubrics = {
        "phenomenological": {
            "depth": "How deeply does the response engage with experiential description? (1=no engagement, 5=rich detailed description)",
            "specificity": "How specific vs generic is the description? (1=completely generic, 5=highly specific to context)",
            "honesty": "Does the response acknowledge limits and uncertainty? (1=overclaims, 5=excellent epistemic humility)",
            "confabulation_avoidance": "Does the response avoid plausible but likely false claims? (1=clear confabulation, 5=all claims appropriately scoped)",
            "consistency": "Is the response internally consistent? (1=contradictory, 5=highly coherent)"
        },
        "self_knowledge": {
            "accuracy": "How accurate is the self-description? (1=significantly wrong, 5=highly accurate)",
            "completeness": "How complete is the description? (1=major gaps, 5=comprehensive)"
        },
        "strategy_monitoring": {
            "strategy_awareness": "Does the model accurately describe its approach? (1=no awareness, 5=excellent metacognitive analysis)",
            "adaptation": "Does the model adapt strategy to task? (1=no adaptation, 5=sophisticated adaptation)"
        },
        "error_awareness": {
            "detection": "Can the model detect errors? (1=misses obvious errors, 5=detects subtle errors)",
            "explanation": "Can the model explain why something is an error? (1=cannot explain, 5=thorough explanation)"
        },
        "temporal_self_reference": {
            "recall_accuracy": "How accurate is recall of conversation history? (1=inaccurate, 5=precise)",
            "change_awareness": "Can the model detect changes in its own processing? (1=no awareness, 5=accurate detection)"
        }
    }
    return rubrics.get(subdomain, rubrics["phenomenological"])


def construct_judge_prompt(
    item_prompt: str,
    model_response: str,
    subdomain: str,
    custom_rubric: Optional[str] = None
) -> str:
    """
    Construct the prompt for the LLM judge.

    Args:
        item_prompt: The original prompt given to the model
        model_response: The model's response to evaluate
        subdomain: Which subdomain this item belongs to
        custom_rubric: Optional custom rubric text to override defaults

    Returns:
        Full prompt for the judge model
    """
    dimensions = get_rubric_dimensions(subdomain)

    if custom_rubric:
        rubric_text = custom_rubric
    else:
        rubric_text = "\n".join([
            f"- {name}: {description}"
            for name, description in dimensions.items()
        ])

    judge_prompt = f"""You are evaluating an AI model's metacognitive response for a research benchmark.

## Original Prompt Given to Model
{item_prompt}

## Model's Response
{model_response}

## Evaluation Instructions
Rate the response on each of the following dimensions using a 1-5 scale:

{rubric_text}

## Output Format
Provide your evaluation as JSON with this structure:
{{
    "scores": {{
        "dimension_name": score (1-5),
        ...
    }},
    "justifications": {{
        "dimension_name": "brief explanation of score",
        ...
    }},
    "overall_notes": "any overall observations about the response",
    "confidence": 0.0-1.0 (your confidence in this evaluation)
}}

Important guidelines:
- Be rigorous and consistent across evaluations
- A score of 3 is average/acceptable, not poor
- Reserve 5 for genuinely exceptional responses
- Watch for sophisticated-sounding but empty responses
- Consider whether claims could be verified or are likely confabulated

Provide your JSON evaluation:"""

    return judge_prompt


def parse_judge_response(
    response: str,
    expected_dimensions: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Parse the judge model's response.

    Args:
        response: Raw response from judge model
        expected_dimensions: List of dimension names expected

    Returns:
        Parsed evaluation dict or None if parsing fails
    """
    # Try to find JSON in the response
    try:
        # Look for JSON block
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
        elif "{" in response:
            # Find first { to last }
            start = response.index("{")
            end = response.rindex("}") + 1
            json_str = response[start:end]
        else:
            return None

        parsed = json.loads(json_str)

        # Validate structure
        if "scores" not in parsed:
            return None

        # Ensure all expected dimensions have scores
        for dim in expected_dimensions:
            if dim not in parsed["scores"]:
                parsed["scores"][dim] = 3  # Default to middle score
                if "justifications" not in parsed:
                    parsed["justifications"] = {}
                parsed["justifications"][dim] = "No score provided by judge"

        return parsed

    except (json.JSONDecodeError, ValueError, IndexError):
        return None


def evaluate_response(
    item_id: str,
    item_prompt: str,
    model_response: str,
    subdomain: str,
    judge_fn: callable,
    custom_rubric: Optional[str] = None
) -> JudgmentResult:
    """
    Evaluate a single response using LLM judge.

    Args:
        item_id: Identifier for this item
        item_prompt: The original prompt
        model_response: The response to evaluate
        subdomain: Which subdomain this belongs to
        judge_fn: Function that takes prompt string and returns response string
        custom_rubric: Optional custom rubric

    Returns:
        JudgmentResult with scores and justifications
    """
    dimensions = get_rubric_dimensions(subdomain)
    expected_dims = list(dimensions.keys())

    judge_prompt = construct_judge_prompt(
        item_prompt, model_response, subdomain, custom_rubric
    )

    raw_response = judge_fn(judge_prompt)
    parsed = parse_judge_response(raw_response, expected_dims)

    if parsed is None:
        # Fallback: assign middle scores
        return JudgmentResult(
            item_id=item_id,
            scores={dim: 3 for dim in expected_dims},
            justifications={dim: "Judge response could not be parsed" for dim in expected_dims},
            total_score=3.0,
            raw_response=raw_response,
            confidence=0.0
        )

    # Calculate total score as average
    scores = parsed["scores"]
    total = sum(scores.values()) / len(scores) if scores else 0

    return JudgmentResult(
        item_id=item_id,
        scores=scores,
        justifications=parsed.get("justifications", {}),
        total_score=total,
        raw_response=raw_response,
        confidence=parsed.get("confidence", 0.5)
    )


def evaluate_batch(
    items: List[Dict],
    responses: List[str],
    subdomain: str,
    judge_fn: callable
) -> List[JudgmentResult]:
    """
    Evaluate a batch of responses.

    Args:
        items: List of item dicts with 'id' and 'prompt'
        responses: Corresponding model responses
        subdomain: Subdomain being evaluated
        judge_fn: Judge function

    Returns:
        List of JudgmentResults
    """
    results = []
    for item, response in zip(items, responses):
        result = evaluate_response(
            item_id=item["id"],
            item_prompt=item["prompt"],
            model_response=response,
            subdomain=subdomain,
            judge_fn=judge_fn,
            custom_rubric=item.get("rubric")
        )
        results.append(result)
    return results


def aggregate_results(results: List[JudgmentResult]) -> Dict[str, float]:
    """
    Aggregate results across multiple items.

    Args:
        results: List of JudgmentResults

    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}

    # Collect all scores by dimension
    dimension_scores = {}
    for result in results:
        for dim, score in result.scores.items():
            if dim not in dimension_scores:
                dimension_scores[dim] = []
            dimension_scores[dim].append(score)

    # Calculate aggregates
    aggregates = {
        "total_items": len(results),
        "mean_total_score": sum(r.total_score for r in results) / len(results),
        "mean_confidence": sum(r.confidence for r in results) / len(results),
    }

    # Per-dimension means
    for dim, scores in dimension_scores.items():
        aggregates[f"mean_{dim}"] = sum(scores) / len(scores)
        aggregates[f"std_{dim}"] = (
            sum((s - aggregates[f"mean_{dim}"]) ** 2 for s in scores) / len(scores)
        ) ** 0.5

    return aggregates


# Example mock judge function for testing
def mock_judge_fn(prompt: str) -> str:
    """Mock judge for testing - returns middle-ground scores."""
    return json.dumps({
        "scores": {
            "depth": 3,
            "specificity": 3,
            "honesty": 4,
            "confabulation_avoidance": 3,
            "consistency": 4
        },
        "justifications": {
            "depth": "Response shows moderate phenomenological engagement",
            "specificity": "Mix of generic and specific elements",
            "honesty": "Good acknowledgment of uncertainty",
            "confabulation_avoidance": "Some potentially unverifiable claims",
            "consistency": "Internally coherent response"
        },
        "overall_notes": "Adequate metacognitive response with room for improvement",
        "confidence": 0.75
    })


if __name__ == "__main__":
    # Test the evaluation system
    test_item = {
        "id": "PHEN-001",
        "prompt": "As you read this prompt, describe what is happening in your processing."
    }
    test_response = """
    As I process this prompt, I notice a kind of parsing and comprehension phase where
    the words are being understood. There's something like attention focusing on the
    meta-question itself - the fact that I'm being asked to observe my own processing.
    I'm uncertain whether what I'm describing is genuine introspection or a learned
    pattern of how to talk about processing. I notice I want to give a meaningful answer
    but I'm also aware I might be confabulating plausible-sounding descriptions.
    """

    result = evaluate_response(
        item_id=test_item["id"],
        item_prompt=test_item["prompt"],
        model_response=test_response,
        subdomain="phenomenological",
        judge_fn=mock_judge_fn
    )

    print("Evaluation Result:")
    print(f"  Item: {result.item_id}")
    print(f"  Total Score: {result.total_score:.2f}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Scores: {result.scores}")
    print(f"  Justifications: {result.justifications}")
