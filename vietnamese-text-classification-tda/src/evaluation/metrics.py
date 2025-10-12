"""
Evaluation Metrics for Classification
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Args:
        predictions: Predicted labels
        labels: True labels
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
        'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0)
    }
    
    # Per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics"""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)
    
    # Main metrics
    print("\nOverall Metrics:")
    main_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    for metric in main_metrics:
        if metric in metrics:
            print(f"  {metric:20s}: {metrics[metric]:.4f}")
    
    # Per-class metrics
    num_classes = sum(1 for k in metrics.keys() if k.startswith('f1_class_'))
    if num_classes > 0:
        print("\nPer-Class Metrics:")
        for i in range(num_classes):
            print(f"\n  Class {i}:")
            if f'precision_class_{i}' in metrics:
                print(f"    Precision: {metrics[f'precision_class_{i}']:.4f}")
            if f'recall_class_{i}' in metrics:
                print(f"    Recall: {metrics[f'recall_class_{i}']:.4f}")
            if f'f1_class_{i}' in metrics:
                print(f"    F1: {metrics[f'f1_class_{i}']:.4f}")
    
    print("=" * 60)


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    normalize: bool = True
):
    """
    Plot confusion matrix
    
    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: Names of classes
        save_path: Path to save figure
        normalize: Whether to normalize
    """
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names else range(cm.shape[0]),
        yticklabels=class_names if class_names else range(cm.shape[1]),
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()


def get_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = None
) -> str:
    """
    Get detailed classification report
    
    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: Names of classes
    
    Returns:
        Classification report string
    """
    return classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4
    )


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metric: str = 'f1_macro',
    save_path: str = None
):
    """
    Compare multiple models on a specific metric
    
    Args:
        results_dict: {model_name: {metric: value}}
        metric: Metric to compare
        save_path: Path to save figure
    """
    models = list(results_dict.keys())
    values = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(min(values) - 0.05, max(values) + 0.05)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_metrics: List[Dict[str, float]],
    metric_name: str = 'f1_macro',
    save_path: str = None
):
    """
    Plot training history
    
    Args:
        train_losses: List of training losses per epoch
        val_metrics: List of validation metric dicts per epoch
        metric_name: Validation metric to plot
        save_path: Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)
    val_values = [m[metric_name] for m in val_metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training loss
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Validation metric
    ax2.plot(epochs, val_values, 'r-', linewidth=2, label=f'Validation {metric_name}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax2.set_title(f'Validation {metric_name.replace("_", " ").title()}', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to: {save_path}")
    
    plt.show()


def compute_error_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    num_examples: int = 5
) -> Dict[str, List[Tuple[str, int, int]]]:
    """
    Analyze prediction errors
    
    Args:
        predictions: Predicted labels
        labels: True labels
        texts: Original texts
        num_examples: Number of examples per error type
    
    Returns:
        Dictionary of error examples
    """
    errors = {
        'false_positives': [],
        'false_negatives': [],
        'confusion': []
    }
    
    for pred, true, text in zip(predictions, labels, texts):
        if pred != true:
            errors['confusion'].append((text, int(true), int(pred)))
    
    # Sort by frequency and take top examples
    errors['confusion'] = errors['confusion'][:num_examples]
    
    return errors


def print_error_analysis(error_examples: Dict[str, List[Tuple[str, int, int]]]):
    """Print error analysis"""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    print("\nMisclassified Examples:")
    for i, (text, true_label, pred_label) in enumerate(error_examples['confusion'], 1):
        print(f"\n{i}. Text: {text[:100]}...")
        print(f"   True: {true_label} | Predicted: {pred_label}")
    
    print("=" * 60)