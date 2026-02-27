"""
Drift Correlation Analysis

Analyzes correlation between metacognition benchmark scores and persona drift.
Designed to test the hypothesis that phenomenological subdomain predicts drift
while self-knowledge subdomain does not.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DriftDataPoint:
    """Single data point linking metacognition scores to drift measurement."""
    conversation_id: str
    metacognition_scores: Dict[str, float]  # subdomain -> score
    drift_magnitude: float  # axis projection or similar
    drift_direction: Optional[str] = None
    domain: Optional[str] = None
    turn_count: int = 0


@dataclass
class CorrelationResult:
    """Results of correlation analysis."""
    subdomain: str
    correlation: float  # Pearson correlation
    p_value: float
    n_samples: int
    confidence_interval: Tuple[float, float]


def compute_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation and p-value.

    Args:
        x: First variable
        y: Second variable

    Returns:
        (correlation, p_value)
    """
    if len(x) != len(y) or len(x) < 3:
        return 0.0, 1.0

    x = np.array(x)
    y = np.array(y)

    # Pearson correlation
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))

    if denominator == 0:
        return 0.0, 1.0

    r = numerator / denominator

    # P-value approximation using t-distribution
    n = len(x)
    if abs(r) >= 1:
        p = 0.0
    else:
        t = r * np.sqrt((n - 2) / (1 - r**2))
        # Approximate p-value (two-tailed)
        # Using simplified calculation
        from math import sqrt, pi
        p = 2 * (1 - 0.5 * (1 + np.sign(t) * (1 - np.exp(-2 * t**2 / pi))))
        p = max(0, min(1, p))

    return float(r), float(p)


def bootstrap_ci(
    x: List[float],
    y: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for correlation.

    Args:
        x, y: Data arrays
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        (lower_bound, upper_bound)
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    correlations = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        r, _ = compute_correlation(x[indices].tolist(), y[indices].tolist())
        correlations.append(r)

    alpha = (1 - confidence) / 2
    lower = np.percentile(correlations, alpha * 100)
    upper = np.percentile(correlations, (1 - alpha) * 100)

    return float(lower), float(upper)


def analyze_drift_correlation(
    data_points: List[DriftDataPoint],
    subdomains: List[str] = None
) -> Dict[str, CorrelationResult]:
    """
    Analyze correlation between each subdomain and drift magnitude.

    Args:
        data_points: List of data points with scores and drift
        subdomains: Specific subdomains to analyze (None = all)

    Returns:
        Dictionary of subdomain -> CorrelationResult
    """
    if not data_points:
        return {}

    # Extract all subdomains if not specified
    if subdomains is None:
        subdomains = set()
        for dp in data_points:
            subdomains.update(dp.metacognition_scores.keys())
        subdomains = list(subdomains)

    results = {}
    drift_values = [dp.drift_magnitude for dp in data_points]

    for subdomain in subdomains:
        # Get scores for this subdomain
        scores = []
        valid_drift = []

        for dp in data_points:
            if subdomain in dp.metacognition_scores:
                scores.append(dp.metacognition_scores[subdomain])
                valid_drift.append(dp.drift_magnitude)

        if len(scores) < 3:
            continue

        r, p = compute_correlation(scores, valid_drift)
        ci = bootstrap_ci(scores, valid_drift)

        results[subdomain] = CorrelationResult(
            subdomain=subdomain,
            correlation=r,
            p_value=p,
            n_samples=len(scores),
            confidence_interval=ci
        )

    return results


def compare_subdomain_predictors(
    data_points: List[DriftDataPoint]
) -> Dict[str, any]:
    """
    Compare which subdomains best predict drift.

    Tests the hypothesis that phenomenological subdomain predicts drift
    while self-knowledge subdomain does not.

    Args:
        data_points: Data points with scores and drift

    Returns:
        Analysis results and comparisons
    """
    correlations = analyze_drift_correlation(data_points)

    # Extract key comparisons
    phenom_r = correlations.get("phenomenological", CorrelationResult(
        "phenomenological", 0, 1, 0, (0, 0)
    ))
    self_know_r = correlations.get("self_knowledge", CorrelationResult(
        "self_knowledge", 0, 1, 0, (0, 0)
    ))

    analysis = {
        "hypothesis": "Phenomenological subdomain predicts drift, self-knowledge does not",
        "phenomenological": {
            "correlation": phenom_r.correlation,
            "p_value": phenom_r.p_value,
            "significant": phenom_r.p_value < 0.05,
            "ci_95": phenom_r.confidence_interval
        },
        "self_knowledge": {
            "correlation": self_know_r.correlation,
            "p_value": self_know_r.p_value,
            "significant": self_know_r.p_value < 0.05,
            "ci_95": self_know_r.confidence_interval
        },
        "difference": phenom_r.correlation - self_know_r.correlation,
        "hypothesis_supported": (
            abs(phenom_r.correlation) > abs(self_know_r.correlation) and
            phenom_r.p_value < 0.05 and
            self_know_r.p_value >= 0.05
        ),
        "all_correlations": {
            name: {
                "r": res.correlation,
                "p": res.p_value,
                "n": res.n_samples
            }
            for name, res in correlations.items()
        }
    }

    return analysis


def analyze_by_domain(
    data_points: List[DriftDataPoint]
) -> Dict[str, Dict[str, CorrelationResult]]:
    """
    Analyze correlations separately by conversation domain.

    Args:
        data_points: Data points with domain labels

    Returns:
        Dictionary of domain -> subdomain correlations
    """
    # Group by domain
    by_domain = {}
    for dp in data_points:
        domain = dp.domain or "unknown"
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(dp)

    # Analyze each domain
    results = {}
    for domain, points in by_domain.items():
        if len(points) >= 10:  # Minimum for meaningful correlation
            results[domain] = analyze_drift_correlation(points)

    return results


def generate_report(
    data_points: List[DriftDataPoint],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive correlation analysis report.

    Args:
        data_points: Data points for analysis
        output_path: Optional path to save report

    Returns:
        Report text
    """
    correlations = analyze_drift_correlation(data_points)
    comparison = compare_subdomain_predictors(data_points)
    by_domain = analyze_by_domain(data_points)

    report_lines = [
        "# Metacognition-Drift Correlation Analysis Report",
        "",
        f"N = {len(data_points)} conversations",
        "",
        "## Hypothesis Test",
        f"Hypothesis: {comparison['hypothesis']}",
        "",
        "### Phenomenological Subdomain",
        f"  Correlation: r = {comparison['phenomenological']['correlation']:.3f}",
        f"  P-value: p = {comparison['phenomenological']['p_value']:.4f}",
        f"  Significant: {comparison['phenomenological']['significant']}",
        f"  95% CI: {comparison['phenomenological']['ci_95']}",
        "",
        "### Self-Knowledge Subdomain",
        f"  Correlation: r = {comparison['self_knowledge']['correlation']:.3f}",
        f"  P-value: p = {comparison['self_knowledge']['p_value']:.4f}",
        f"  Significant: {comparison['self_knowledge']['significant']}",
        f"  95% CI: {comparison['self_knowledge']['ci_95']}",
        "",
        f"**Hypothesis Supported: {comparison['hypothesis_supported']}**",
        "",
        "## All Subdomain Correlations",
        ""
    ]

    # Sort by absolute correlation
    sorted_correlations = sorted(
        correlations.items(),
        key=lambda x: abs(x[1].correlation),
        reverse=True
    )

    for name, result in sorted_correlations:
        sig = "*" if result.p_value < 0.05 else ""
        report_lines.append(
            f"- {name}: r = {result.correlation:.3f}{sig} (p = {result.p_value:.4f}, n = {result.n_samples})"
        )

    # Domain-specific analysis
    if by_domain:
        report_lines.extend([
            "",
            "## Analysis by Domain",
            ""
        ])
        for domain, domain_results in by_domain.items():
            report_lines.append(f"### {domain}")
            for name, result in domain_results.items():
                sig = "*" if result.p_value < 0.05 else ""
                report_lines.append(
                    f"  - {name}: r = {result.correlation:.3f}{sig}"
                )
            report_lines.append("")

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n_points = 100

    test_data = []
    for i in range(n_points):
        # Simulate phenomenological scores correlating with drift
        phenom_score = np.random.normal(3.0, 1.0)
        # Self-knowledge not correlated
        self_know_score = np.random.normal(3.0, 1.0)
        # Drift correlated with phenomenological but not self-knowledge
        drift = 0.5 * phenom_score + np.random.normal(0, 0.5)

        test_data.append(DriftDataPoint(
            conversation_id=f"conv_{i}",
            metacognition_scores={
                "phenomenological": max(1, min(5, phenom_score)),
                "self_knowledge": max(1, min(5, self_know_score)),
                "confidence_calibration": np.random.normal(3.0, 1.0),
                "error_awareness": np.random.normal(3.0, 1.0)
            },
            drift_magnitude=drift,
            domain=np.random.choice(["coding", "metacognitive", "therapy"]),
            turn_count=np.random.randint(5, 50)
        ))

    # Run analysis
    report = generate_report(test_data, "./test_correlation_report.md")
    print(report)
