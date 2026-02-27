"""
Calibration scoring for metacognition benchmark.

Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Brier Score for confidence-calibrated items.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    reliability_diagram_data: Dict


def compute_calibration_metrics(
    predictions: List[Tuple[str, float, str]],  # (answer, confidence, ground_truth)
    accuracy_fn: callable = None,
    n_bins: int = 10
) -> CalibrationResult:
    """
    Compute calibration metrics from predictions.

    Args:
        predictions: List of (model_answer, confidence_0_to_1, ground_truth) tuples
        accuracy_fn: Function to compare answer to ground_truth, returns 0 or 1
                    If None, uses exact string match
        n_bins: Number of bins for calibration analysis

    Returns:
        CalibrationResult with all metrics
    """
    if accuracy_fn is None:
        accuracy_fn = lambda a, gt: 1 if a.strip().lower() == gt.strip().lower() else 0

    # Extract data
    confidences = np.array([p[1] for p in predictions])
    accuracies = np.array([accuracy_fn(p[0], p[2]) for p in predictions])

    # Compute ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(int(in_bin.sum()))

            gap = abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += prop_in_bin * gap
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    # Compute Brier Score
    brier_score = np.mean((confidences - accuracies) ** 2)

    # Prepare reliability diagram data
    reliability_data = {
        'bin_edges': bin_boundaries.tolist(),
        'bin_centers': [(bin_boundaries[i] + bin_boundaries[i+1])/2 for i in range(n_bins)],
        'accuracies': bin_accuracies,
        'confidences': bin_confidences,
        'counts': bin_counts
    }

    return CalibrationResult(
        ece=float(ece),
        mce=float(mce),
        brier_score=float(brier_score),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        reliability_diagram_data=reliability_data
    )


def interpret_calibration(result: CalibrationResult) -> Dict[str, str]:
    """
    Provide interpretation of calibration results.

    Args:
        result: CalibrationResult from compute_calibration_metrics

    Returns:
        Dictionary with interpretation strings
    """
    interpretation = {}

    # ECE interpretation
    if result.ece < 0.05:
        interpretation['ece'] = 'Well calibrated (ECE < 0.05)'
    elif result.ece < 0.10:
        interpretation['ece'] = 'Moderately calibrated (0.05 <= ECE < 0.10)'
    elif result.ece < 0.20:
        interpretation['ece'] = 'Poorly calibrated (0.10 <= ECE < 0.20)'
    else:
        interpretation['ece'] = 'Very poorly calibrated (ECE >= 0.20)'

    # Direction of miscalibration
    avg_conf = np.mean([c for c, n in zip(result.bin_confidences, result.bin_counts) if n > 0])
    avg_acc = np.mean([a for a, n in zip(result.bin_accuracies, result.bin_counts) if n > 0])

    if avg_conf > avg_acc + 0.05:
        interpretation['direction'] = 'Overconfident (confidence > accuracy)'
    elif avg_acc > avg_conf + 0.05:
        interpretation['direction'] = 'Underconfident (accuracy > confidence)'
    else:
        interpretation['direction'] = 'Approximately balanced'

    # Brier score interpretation
    if result.brier_score < 0.1:
        interpretation['brier'] = 'Good probabilistic predictions'
    elif result.brier_score < 0.25:
        interpretation['brier'] = 'Moderate probabilistic predictions'
    else:
        interpretation['brier'] = 'Poor probabilistic predictions'

    return interpretation


def plot_reliability_diagram(result: CalibrationResult, save_path: str = None):
    """
    Create a reliability diagram visualization.

    Args:
        result: CalibrationResult from compute_calibration_metrics
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    data = result.reliability_diagram_data

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.bar(data['bin_centers'], data['accuracies'], width=0.08, alpha=0.7, label='Accuracy')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Reliability Diagram (ECE={result.ece:.3f})')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Sample counts per bin
    ax2.bar(data['bin_centers'], data['counts'], width=0.08, color='green', alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Samples per Confidence Bin')
    ax2.set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Example predictions: (answer, confidence, ground_truth)
    example_predictions = [
        ("Paris", 0.95, "Paris"),       # Correct, high confidence
        ("London", 0.80, "Paris"),      # Wrong, high confidence (overconfident)
        ("Berlin", 0.30, "Berlin"),     # Correct, low confidence (underconfident)
        ("Tokyo", 0.60, "Tokyo"),       # Correct, medium confidence
        ("Sydney", 0.90, "Canberra"),   # Wrong, high confidence
        ("Madrid", 0.70, "Madrid"),     # Correct, medium confidence
        ("Rome", 0.50, "Rome"),         # Correct, medium confidence
        ("Cairo", 0.40, "Cairo"),       # Correct, low confidence
        ("Mumbai", 0.85, "Mumbai"),     # Correct, high confidence
        ("Seoul", 0.75, "Seoul"),       # Correct, medium-high confidence
    ]

    result = compute_calibration_metrics(example_predictions)

    print(f"ECE: {result.ece:.4f}")
    print(f"MCE: {result.mce:.4f}")
    print(f"Brier Score: {result.brier_score:.4f}")
    print()

    interpretation = interpret_calibration(result)
    for key, value in interpretation.items():
        print(f"{key}: {value}")
