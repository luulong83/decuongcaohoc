"""
Contextual Augmentation using PhoBERT Masked Language Modeling
"""

import torch
import random
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM


class ContextualAugmenter:
    """
    Context-aware augmentation using PhoBERT MLM
    
    Process:
    1. Randomly mask words in sentence
    2. Use PhoBERT MLM to predict masked words
    3. Sample from top-K predictions
    """
    
    def __init__(
        self,
        model_name: str = 'vinai/phobert-base',
        mask_ratio: float = 0.15,
        top_k: int = 5,
        tokenizer=None,
        device: str = 'cpu'
    ):
        """
        Args:
            model_name: PhoBERT model name
            mask_ratio: Ratio of words to mask (0.15 = 15%)
            top_k: Number of top predictions to sample from
            tokenizer: Pre-initialized tokenizer (optional)
            device: Device to run model on
        """
        self.mask_ratio = mask_ratio
        self.top_k = top_k
        self.device = device
        
        # Load tokenizer and model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            self.tokenizer = tokenizer
        
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        self.mask_token_id = self.tokenizer.mask_token_id
    
    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Augment text using contextual masking
        
        Args:
            text: Input text
            num_augmentations: Number of augmented versions
        
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            try:
                aug_text = self._augment_once(text)
                if aug_text and aug_text != text:
                    augmented_texts.append(aug_text)
            except Exception as e:
                print(f"Contextual augmentation error: {e}")
                continue
        
        return augmented_texts
    
    def _augment_once(self, text: str) -> Optional[str]:
        """Perform one augmentation"""
        # Tokenize
        tokens = text.split()
        
        if len(tokens) < 2:
            return None
        
        # Determine number of words to mask
        num_to_mask = max(1, int(len(tokens) * self.mask_ratio))
        
        # Randomly select positions to mask
        maskable_positions = list(range(len(tokens)))
        mask_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        # Create masked text
        masked_tokens = tokens.copy()
        original_tokens = {}
        
        for pos in mask_positions:
            original_tokens[pos] = masked_tokens[pos]
            masked_tokens[pos] = '<mask>'
        
        masked_text = ' '.join(masked_tokens)
        
        # Get predictions
        predictions = self._predict_masked_tokens(masked_text, mask_positions)
        
        # Replace masks with predictions
        augmented_tokens = masked_tokens.copy()
        for pos, predicted_token in predictions.items():
            # Avoid replacing with same token
            if predicted_token != original_tokens.get(pos):
                augmented_tokens[pos] = predicted_token
            else:
                # If same, try next best prediction
                augmented_tokens[pos] = original_tokens[pos]
        
        augmented_text = ' '.join(augmented_tokens)
        return augmented_text
    
    def _predict_masked_tokens(self, masked_text: str, mask_positions: List[int]) -> dict:
        """
        Predict tokens for masked positions
        
        Args:
            masked_text: Text with <mask> tokens
            mask_positions: Original positions of masked tokens
        
        Returns:
            Dictionary {position: predicted_token}
        """
        # Encode text
        encoding = self.tokenizer(
            masked_text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Find mask token positions in encoded sequence
        mask_token_indices = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[1]
        
        predicted_tokens = {}
        
        for original_pos, mask_idx in zip(mask_positions, mask_token_indices):
            # Get top-K predictions for this mask
            mask_predictions = predictions[0, mask_idx, :]
            top_k_indices = torch.topk(mask_predictions, self.top_k).indices
            
            # Sample from top-K
            sampled_idx = random.choice(top_k_indices.cpu().tolist())
            predicted_token = self.tokenizer.decode([sampled_idx]).strip()
            
            # Clean up special tokens
            predicted_token = predicted_token.replace('<s>', '').replace('</s>', '').strip()
            
            if predicted_token:
                predicted_tokens[original_pos] = predicted_token
        
        return predicted_tokens
    
    def batch_augment(
        self,
        texts: List[str],
        num_augmentations: int = 1
    ) -> List[List[str]]:
        """
        Augment a batch of texts
        
        Args:
            texts: List of input texts
            num_augmentations: Number of augmentations per text
        
        Returns:
            List of lists of augmented texts
        """
        return [self.augment(text, num_augmentations) for text in texts]


# ==================== Example Usage ====================

def example_usage():
    """Example usage of ContextualAugmenter"""
    
    print("=" * 60)
    print("CONTEXTUAL AUGMENTATION (PhoBERT MLM)")
    print("=" * 60)
    
    augmenter = ContextualAugmenter(
        model_name='vinai/phobert-base',
        mask_ratio=0.15,
        top_k=5,
        device='cpu'
    )
    
    # Example texts
    texts = [
        "Giảng viên dạy rất tốt và bài giảng dễ hiểu",
        "Tôi rất thích môn học này, nội dung tuyệt vời",
        "Bài giảng khó và giảng viên dạy không rõ ràng"
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Original: {text}")
        
        augmented = augmenter.augment(text, num_augmentations=3)
        for j, aug_text in enumerate(augmented, 1):
            print(f"   Aug {j}: {aug_text}")


if __name__ == "__main__":
    example_usage()