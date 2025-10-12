"""
UIT-VSFC Dataset Loader
Handles Vietnamese Students' Feedback Corpus
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

from src.data.augmentation.synonym_replacement import VietnameseSynonymReplacer
from src.data.augmentation.back_translation import CachedBackTranslator
from src.data.augmentation.contextual_aug import ContextualAugmenter


class UITVSFCDataset(Dataset):
    """
    UIT-VSFC Dataset for sentiment and topic classification
    
    Dataset structure:
    - train/dev/test splits
    - sentiments.txt: Labels (Positive/Neutral/Negative)
    - topics.txt: Labels (10 topics)
    - sents.txt: Text sentences
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
        task: str = 'sentiment'
    ):
        """
        Args:
            texts: List of text sentences
            labels: List of integer labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            task: 'sentiment' or 'topic'
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
        assert len(texts) == len(labels), "Texts and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text  # Keep original text for debugging
        }


def load_uitvsfc_split(
    data_dir: str,
    split: str,
    task: str = 'sentiment'
) -> Tuple[List[str], List[int]]:
    """
    Load one split of UIT-VSFC dataset
    
    Args:
        data_dir: Path to data directory
        split: 'train', 'dev', or 'test'
        task: 'sentiment' or 'topic'
    
    Returns:
        (texts, labels) tuple
    """
    split_dir = Path(data_dir) / split
    
    # Read texts
    sents_file = split_dir / 'sents.txt'
    with open(sents_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Read labels
    if task == 'sentiment':
        labels_file = split_dir / 'sentiments.txt'
        label_map = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
    else:  # topic
        labels_file = split_dir / 'topics.txt'
        # Topic labels are already integers in the file
        label_map = None
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        if label_map:
            labels = [label_map[line.strip()] for line in f if line.strip()]
        else:
            labels = [int(line.strip()) for line in f if line.strip()]
    
    assert len(texts) == len(labels), f"Mismatch in {split}: {len(texts)} texts vs {len(labels)} labels"
    
    return texts, labels


def augment_dataset(
    texts: List[str],
    labels: List[int],
    augmentation_config: Dict[str, Any],
    tokenizer
) -> Tuple[List[str], List[int]]:
    """
    Augment dataset using multiple techniques
    
    Args:
        texts: Original texts
        labels: Original labels
        augmentation_config: Augmentation configuration
        tokenizer: Tokenizer for contextual augmentation
    
    Returns:
        (augmented_texts, augmented_labels) including originals
    """
    if not augmentation_config['enabled']:
        return texts, labels
    
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    
    total_ratio = augmentation_config['ratio']
    num_to_augment = int(len(texts) * total_ratio)
    
    # Initialize augmenters
    augmenters = {}
    
    if augmentation_config['techniques']['synonym_replacement']['enabled']:
        augmenters['sr'] = VietnameseSynonymReplacer(
            replace_ratio=augmentation_config['techniques']['synonym_replacement']['replace_ratio'],
            tone_aware=augmentation_config['techniques']['synonym_replacement']['tone_aware'],
            preserve_compounds=augmentation_config['techniques']['synonym_replacement']['preserve_compounds']
        )
    
    if augmentation_config['techniques']['back_translation']['enabled']:
        augmenters['bt'] = CachedBackTranslator(
            quality_threshold=augmentation_config['techniques']['back_translation']['quality_threshold'],
            max_bleu=augmentation_config['techniques']['back_translation']['max_bleu']
        )
    
    if augmentation_config['techniques']['contextual']['enabled']:
        augmenters['ctx'] = ContextualAugmenter(
            model_name=augmentation_config['techniques']['contextual']['model'],
            mask_ratio=augmentation_config['techniques']['contextual']['mask_ratio'],
            top_k=augmentation_config['techniques']['contextual']['top_k'],
            tokenizer=tokenizer
        )
    
    # Compute number of augmentations per technique
    weights = {}
    if 'sr' in augmenters:
        weights['sr'] = augmentation_config['techniques']['synonym_replacement']['weight']
    if 'bt' in augmenters:
        weights['bt'] = augmentation_config['techniques']['back_translation']['weight']
    if 'ctx' in augmenters:
        weights['ctx'] = augmentation_config['techniques']['contextual']['weight']
    
    # Normalize weights
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight
    
    # Sample texts to augment
    indices_to_augment = np.random.choice(len(texts), num_to_augment, replace=False)
    
    print(f"\n🔄 Augmenting {num_to_augment} samples...")
    
    for idx in indices_to_augment:
        text = texts[idx]
        label = labels[idx]
        
        # Randomly choose augmentation technique
        technique = np.random.choice(list(weights.keys()), p=list(weights.values()))
        
        # Apply augmentation
        try:
            if technique == 'sr':
                aug_texts = augmenters['sr'].augment(text, num_augmentations=1)
            elif technique == 'bt':
                aug_texts = augmenters['bt'].augment(text, num_augmentations=1)
            elif technique == 'ctx':
                aug_texts = augmenters['ctx'].augment(text, num_augmentations=1)
            else:
                aug_texts = []
            
            # Add augmented samples
            for aug_text in aug_texts:
                if aug_text and aug_text != text:  # Avoid duplicates
                    augmented_texts.append(aug_text)
                    augmented_labels.append(label)
        
        except Exception as e:
            print(f"Warning: Augmentation failed for text: {text[:50]}... Error: {e}")
            continue
    
    print(f"✅ Augmentation complete: {len(texts)} → {len(augmented_texts)} samples")
    
    return augmented_texts, augmented_labels


def load_uitvsfc_data(
    data_dir: str,
    task: str = 'sentiment',
    max_length: int = 256,
    use_augmentation: bool = False,
    augmentation_config: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Tuple[UITVSFCDataset, UITVSFCDataset, UITVSFCDataset]:
    """
    Load complete UIT-VSFC dataset with train/dev/test splits
    
    Args:
        data_dir: Path to data directory
        task: 'sentiment' or 'topic'
        max_length: Maximum sequence length
        use_augmentation: Whether to augment training data
        augmentation_config: Configuration for augmentation
        debug: If True, use small subset for debugging
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    print(f"\n📊 Loading UIT-VSFC dataset (task: {task})...")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
    
    # Load splits
    train_texts, train_labels = load_uitvsfc_split(data_dir, 'train', task)
    dev_texts, dev_labels = load_uitvsfc_split(data_dir, 'dev', task)
    test_texts, test_labels = load_uitvsfc_split(data_dir, 'test', task)
    
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Dev: {len(dev_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    # Debug mode - use small subset
    if debug:
        print("⚠️  DEBUG MODE: Using small subset")
        train_texts, train_labels = train_texts[:100], train_labels[:100]
        dev_texts, dev_labels = dev_texts[:50], dev_labels[:50]
        test_texts, test_labels = test_texts[:50], test_labels[:50]
    
    # Augment training data
    if use_augmentation and augmentation_config:
        train_texts, train_labels = augment_dataset(
            train_texts,
            train_labels,
            augmentation_config,
            tokenizer
        )
    
    # Create datasets
    train_dataset = UITVSFCDataset(train_texts, train_labels, tokenizer, max_length, task)
    val_dataset = UITVSFCDataset(dev_texts, dev_labels, tokenizer, max_length, task)
    test_dataset = UITVSFCDataset(test_texts, test_labels, tokenizer, max_length, task)
    
    # Print label distribution
    print(f"\n📊 Label distribution (train):")
    unique, counts = np.unique(train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset