"""
Back Translation for Vietnamese Text Augmentation
Vi → En → Vi pipeline with quality control
"""

import time
from typing import List, Optional, Tuple
from googletrans import Translator
import numpy as np
from difflib import SequenceMatcher


class BackTranslator:
    """
    Back translation augmentation: Vi → En → Vi
    
    Features:
    - Quality control (BLEU-like score)
    - Rate limiting to avoid API blocking
    - Batch processing
    - Error handling
    """
    
    def __init__(
        self,
        source_lang: str = 'vi',
        pivot_lang: str = 'en',
        quality_threshold: float = 0.7,
        max_bleu: float = 0.95,
        rate_limit_delay: float = 0.5
    ):
        """
        Args:
            source_lang: Source language code ('vi')
            pivot_lang: Pivot language for translation ('en')
            quality_threshold: Minimum similarity to original (0.7 = 70%)
            max_bleu: Maximum similarity (avoid too similar translations)
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self.quality_threshold = quality_threshold
        self.max_bleu = max_bleu
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize translator
        self.translator = Translator()
        
        # Statistics
        self.num_successful = 0
        self.num_failed = 0
        self.num_low_quality = 0
    
    def augment(
        self,
        text: str,
        num_augmentations: int = 1
    ) -> List[str]:
        """
        Augment text via back translation
        
        Args:
            text: Input Vietnamese text
            num_augmentations: Number of augmentations (may return fewer due to quality control)
        
        Returns:
            List of back-translated texts
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            try:
                # Perform back translation
                back_translated = self._back_translate_once(text)
                
                if back_translated is None:
                    continue
                
                # Quality check
                quality = self._compute_similarity(text, back_translated)
                
                if self.quality_threshold <= quality <= self.max_bleu:
                    augmented_texts.append(back_translated)
                    self.num_successful += 1
                else:
                    self.num_low_quality += 1
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            except Exception as e:
                print(f"Back translation error: {e}")
                self.num_failed += 1
                continue
        
        return augmented_texts
    
    def _back_translate_once(self, text: str) -> Optional[str]:
        """
        Perform one back translation
        
        Args:
            text: Input text
        
        Returns:
            Back-translated text or None if failed
        """
        try:
            # Step 1: Vi → En
            translation = self.translator.translate(
                text,
                src=self.source_lang,
                dest=self.pivot_lang
            )
            
            if translation is None or not hasattr(translation, 'text'):
                return None
            
            pivot_text = translation.text
            
            # Small delay
            time.sleep(0.1)
            
            # Step 2: En → Vi
            back_translation = self.translator.translate(
                pivot_text,
                src=self.pivot_lang,
                dest=self.source_lang
            )
            
            if back_translation is None or not hasattr(back_translation, 'text'):
                return None
            
            return back_translation.text
        
        except Exception as e:
            print(f"Translation API error: {e}")
            return None
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts
        
        Uses SequenceMatcher (similar to BLEU but simpler)
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def batch_augment(
        self,
        texts: List[str],
        num_augmentations: int = 1,
        show_progress: bool = True
    ) -> List[List[str]]:
        """
        Augment a batch of texts
        
        Args:
            texts: List of input texts
            num_augmentations: Number of augmentations per text
            show_progress: Whether to show progress
        
        Returns:
            List of lists of augmented texts
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(texts) if show_progress else texts
        
        for text in iterator:
            augmented = self.augment(text, num_augmentations)
            results.append(augmented)
        
        return results
    
    def get_statistics(self) -> dict:
        """Get augmentation statistics"""
        total = self.num_successful + self.num_failed + self.num_low_quality
        return {
            'total_attempts': total,
            'successful': self.num_successful,
            'failed': self.num_failed,
            'low_quality': self.num_low_quality,
            'success_rate': self.num_successful / total if total > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.num_successful = 0
        self.num_failed = 0
        self.num_low_quality = 0


class CachedBackTranslator(BackTranslator):
    """
    Back translator with caching to avoid redundant API calls
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    def _back_translate_once(self, text: str) -> Optional[str]:
        """Back translate with caching"""
        # Check cache
        if text in self.cache:
            return self.cache[text]
        
        # Perform translation
        result = super()._back_translate_once(text)
        
        # Store in cache
        if result is not None:
            self.cache[text] = result
        
        return result
    
    def save_cache(self, filepath: str):
        """Save cache to file"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def load_cache(self, filepath: str):
        """Load cache from file"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            self.cache = json.load(f)


# ==================== Example Usage ====================

def example_usage():
    """Example usage of BackTranslator"""
    
    print("=" * 60)
    print("BACK TRANSLATION AUGMENTATION")
    print("=" * 60)
    print("\nNote: This requires internet connection and may take time")
    print("Recommend using CachedBackTranslator for repeated use\n")
    
    translator = BackTranslator(
        quality_threshold=0.7,
        max_bleu=0.95,
        rate_limit_delay=1.0  # 1 second delay to be safe
    )
    
    # Example texts
    texts = [
        "Giảng viên dạy rất tốt và bài giảng dễ hiểu",
        "Tôi rất thích môn học này"
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"{i}. Original: {text}")
        
        augmented = translator.augment(text, num_augmentations=2)
        
        if augmented:
            for j, aug_text in enumerate(augmented, 1):
                quality = translator._compute_similarity(text, aug_text)
                print(f"   Aug {j}: {aug_text}")
                print(f"   Quality: {quality:.3f}")
        else:
            print("   No successful augmentation")
        
        print()
    
    # Statistics
    stats = translator.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    example_usage()