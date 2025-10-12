"""
Statistical Significance Tests
For comparing model performances
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, List


def paired_ttest(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Paired t-test for comparing two models
    
    Args:
        scores1: Scores from model 1 (e.g., per-sample F1)
        scores2: Scores from model 2
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(scores1, scores2)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Mean difference
    mean_diff = np.mean(scores1 - scores2)
    
    return {
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'alpha': alpha,
        'mean_difference': float(mean_diff),
        'interpretation': (
            f"Model 1 {'significantly' if is_significant else 'not significantly'} "
            f"{'better' if mean_diff > 0 else 'worse'} than Model 2 "
            f"(p={p_value:.4f}, α={alpha})"
        )
    }


def cohens_d(
    scores1: np.ndarray,
    scores2: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Cohen's d effect size
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        threshold: Threshold for "large" effect size
    
    Returns:
        Dictionary with effect size results
    """
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(scores1), np.mean(scores2)
    std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(scores1), len(scores2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    is_large = abs_d > threshold
    
    return {
        'cohens_d': float(d),
        'magnitude': magnitude,
        'is_large_effect': is_large,
        'threshold': threshold,
        'interpretation': (
            f"Effect size is {magnitude} (d={d:.3f}). "
            f"{'Practically significant' if is_large else 'Not practically significant'} "
            f"(threshold={threshold})"
        )
    }


def mcnemar_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    McNemar's test for comparing two models
    
    Tests if two models have significantly different error rates
    
    Args:
        predictions1: Predictions from model 1
        predictions2: Predictions from model 2
        labels: True labels
    
    Returns:
        Dictionary with test results
    """
    # Create contingency table
    # b = model1 correct, model2 wrong
    # c = model1 wrong, model2 correct
    correct1 = (predictions1 == labels)
    correct2 = (predictions2 == labels)
    
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    
    # McNemar's test statistic
    if b + c == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'is_significant': False,
            'interpretation': "Models make identical errors"
        }
    
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    is_significant = p_value < 0.05
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'b': int(b),
        'c': int(c),
        'interpretation': (
            f"Models {'have' if is_significant else 'do not have'} "
            f"significantly different error rates (p={p_value:.4f})"
        )
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval
    
    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (0.95 = 95%)
    
    Returns:
        Dictionary with CI results
    """
    bootstrap_means = []
    n = len(scores)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    mean = np.mean(scores)
    
    return {
        'mean': float(mean),
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'interpretation': (
            f"Mean: {mean:.4f}, "
            f"{int(confidence_level*100)}% CI: [{lower:.4f}, {upper:.4f}]"
        )
    }


def compare_multiple_models(
    results_dict: Dict[str, np.ndarray],
    metric_name: str = "F1",
    alpha: float = 0.05
) -> Dict[str, Dict[str, any]]:
    """
    Compare multiple models pairwise
    
    Args:
        results_dict: {model_name: scores_array}
        metric_name: Name of the metric
        alpha: Significance level
    
    Returns:
        Dictionary of pairwise comparisons
    """
    model_names = list(results_dict.keys())
    comparisons = {}
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            key = f"{model1}_vs_{model2}"
            
            scores1 = results_dict[model1]
            scores2 = results_dict[model2]
            
            # Perform tests
            ttest_result = paired_ttest(scores1, scores2, alpha)
            cohens_result = cohens_d(scores1, scores2)
            
            comparisons[key] = {
                'model1': model1,
                'model2': model2,
                'mean1': float(np.mean(scores1)),
                'mean2': float(np.mean(scores2)),
                'ttest': ttest_result,
                'cohens_d': cohens_result
            }
    
    return comparisons


def print_statistical_tests(test_results: Dict[str, any]):
    """Pretty print statistical test results"""
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)
    
    if 'ttest' in test_results:
        print("\n📊 Paired t-test:")
        print(f"  {test_results['ttest']['interpretation']}")
        print(f"  t = {test_results['ttest']['t_statistic']:.4f}")
        print(f"  p = {test_results['ttest']['p_value']:.4f}")
    
    if 'cohens_d' in test_results:
        print("\n📏 Cohen's d (Effect Size):")
        print(f"  {test_results['cohens_d']['interpretation']}")
        print(f"  d = {test_results['cohens_d']['cohens_d']:.4f}")
    
    if 'mcnemar' in test_results:
        print("\n🔀 McNemar's Test:")
        print(f"  {test_results['mcnemar']['interpretation']}")
        print(f"  χ² = {test_results['mcnemar']['statistic']:.4f}")
        print(f"  p = {test_results['mcnemar']['p_value']:.4f}")
    
    if 'bootstrap_ci' in test_results:
        print("\n🎲 Bootstrap Confidence Interval:")
        print(f"  {test_results['bootstrap_ci']['interpretation']}")
    
    print("=" * 60)


def print_pairwise_comparisons(comparisons: Dict[str, Dict[str, any]]):
    """Print pairwise model comparisons"""
    print("\n" + "=" * 60)
    print("PAIRWISE MODEL COMPARISONS")
    print("=" * 60)
    
    for key, result in comparisons.items():
        print(f"\n{result['model1']} vs {result['model2']}:")
        print(f"  Mean 1: {result['mean1']:.4f}")
        print(f"  Mean 2: {result['mean2']:.4f}")
        print(f"  Difference: {result['mean1'] - result['mean2']:.4f}")
        print(f"  Statistically significant: {result['ttest']['is_significant']}")
        print(f"  Effect size: {result['cohens_d']['magnitude']} (d={result['cohens_d']['cohens_d']:.3f})")
    
    print("=" * 60)