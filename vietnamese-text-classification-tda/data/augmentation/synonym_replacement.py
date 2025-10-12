"""
Synonym Replacement for Vietnamese Text
Tone-aware synonym replacement that preserves Vietnamese linguistic features
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from underthesea import word_tokenize
import re


class VietnameseSynonymReplacer:
    """
    Vietnamese synonym replacement with:
    - Tone preservation (ma ≠ mà ≠ má)
    - Compound word handling (máy_bay as single unit)
    - Context awareness
    """
    
    def __init__(
        self,
        replace_ratio: float = 0.1,
        tone_aware: bool = True,
        preserve_compounds: bool = True,
        min_word_length: int = 2
    ):
        """
        Args:
            replace_ratio: Ratio of words to replace (0.1 = 10%)
            tone_aware: Only replace with same-tone synonyms
            preserve_compounds: Treat compound words as single units
            min_word_length: Minimum word length to consider for replacement
        """
        self.replace_ratio = replace_ratio
        self.tone_aware = tone_aware
        self.preserve_compounds = preserve_compounds
        self.min_word_length = min_word_length
        
        # Vietnamese synonym dictionary (simplified - should be expanded)
        self.synonym_dict = self._load_synonym_dict()
        
        # Vietnamese tones
        self.tones = {
            'none': 'aăâeêioôơuưy',
            'acute': 'áắấéếíóốớúứý',  # Sắc
            'grave': 'àằầèềìòồờùừỳ',  # Huyền
            'hook': 'ảẳẩẻểỉỏổởủửỷ',   # Hỏi
            'tilde': 'ãẵẫẽễĩõỗỡũữỹ',  # Ngã
            'dot': 'ạặậẹệịọộợụựỵ'     # Nặng
        }
    
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """
        Load Vietnamese synonym dictionary
        
        In production, this should load from a comprehensive database
        Here we provide a minimal example
        """
        return {
            # Common words
            'tốt': ['hay', 'giỏi', 'xuất_sắc'],
            'xấu': ['dở', 'kém', 'tệ'],
            'đẹp': ['xinh', 'đẹp_đẽ', 'lung_linh'],
            'thích': ['yêu', 'mê', 'ưa'],
            'khó': ['khó_khăn', 'phức_tạp', 'nan_giải'],
            'dễ': ['dễ_dàng', 'đơn_giản', 'nhẹ_nhàng'],
            
            # Education domain
            'giảng_viên': ['thầy', 'cô', 'giáo_viên'],
            'bài_giảng': ['bài_học', 'nội_dung', 'kiến_thức'],
            'học': ['học_hỏi', 'nghiên_cứu', 'tìm_hiểu'],
            'hiểu': ['nắm_bắt', 'tiếp_thu', 'lĩnh_hội'],
            
            # Sentiment words
            'hài_lòng': ['vui', 'vừa_lòng', 'mãn_nguyện'],
            'thất_vọng': ['buồn', 'chán', 'không_vui'],
            'tuyệt_vời': ['tuyệt', 'tốt', 'xuất_sắc', 'tuyệt_hảo'],
            'tệ': ['dở', 'kém', 'tồi', 'tệ_hại'],
        }
    
    def _get_tone(self, word: str) -> str:
        """
        Detect tone of a Vietnamese word
        
        Args:
            word: Vietnamese word
        
        Returns:
            Tone name ('none', 'acute', 'grave', etc.)
        """
        for char in word.lower():
            for tone_name, tone_chars in self.tones.items():
                if char in tone_chars:
                    return tone_name
        return 'none'
    
    def _same_tone(self, word1: str, word2: str) -> bool:
        """Check if two words have the same tone"""
        return self._get_tone(word1) == self._get_tone(word2)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word
        
        Args:
            word: Vietnamese word
        
        Returns:
            List of synonyms (filtered by tone if tone_aware=True)
        """
        word_lower = word.lower()
        
        if word_lower not in self.synonym_dict:
            return []
        
        synonyms = self.synonym_dict[word_lower]
        
        # Filter by tone
        if self.tone_aware:
            synonyms = [s for s in synonyms if self._same_tone(word, s)]
        
        return synonyms
    
    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Augment text by replacing synonyms
        
        Args:
            text: Input text
            num_augmentations: Number of augmented versions to generate
        
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            augmented = self._augment_once(text)
            if augmented != text:  # Only add if actually changed
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def _augment_once(self, text: str) -> str:
        """Perform one augmentation"""
        # Tokenize
        if self.preserve_compounds:
            tokens = word_tokenize(text, format="text").split()
        else:
            tokens = text.split()
        
        # Filter tokens eligible for replacement
        eligible_indices = []
        for i, token in enumerate(tokens):
            if (len(token) >= self.min_word_length and 
                not self._is_special_token(token) and
                self._get_synonyms(token)):
                eligible_indices.append(i)
        
        if not eligible_indices:
            return text
        
        # Determine number of words to replace
        num_to_replace = max(1, int(len(eligible_indices) * self.replace_ratio))
        
        # Randomly select indices to replace
        replace_indices = random.sample(eligible_indices, min(num_to_replace, len(eligible_indices)))
        
        # Perform replacement
        augmented_tokens = tokens.copy()
        for idx in replace_indices:
            word = tokens[idx]
            synonyms = self._get_synonyms(word)
            if synonyms:
                # Randomly choose a synonym
                replacement = random.choice(synonyms)
                augmented_tokens[idx] = replacement
        
        return ' '.join(augmented_tokens)
    
    def _is_special_token(self, token: str) -> bool:
        """Check if token is special (punctuation, numbers, etc.)"""
        # Punctuation
        if token in '.,!?;:"""\'()[]{}':
            return True
        
        # Numbers
        if re.match(r'^\d+$', token):
            return True
        
        # Very short words
        if len(token) < self.min_word_length:
            return True
        
        return False
    
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
    
    def compute_quality_score(self, original: str, augmented: str) -> float:
        """
        Compute quality score of augmentation
        
        Measures how much the text changed (higher = more change)
        
        Args:
            original: Original text
            augmented: Augmented text
        
        Returns:
            Quality score (0-1)
        """
        orig_tokens = set(original.split())
        aug_tokens = set(augmented.split())
        
        if len(orig_tokens) == 0:
            return 0.0
        
        # Jaccard distance (1 - Jaccard similarity)
        intersection = len(orig_tokens & aug_tokens)
        union = len(orig_tokens | aug_tokens)
        
        jaccard_sim = intersection / union if union > 0 else 0
        jaccard_dist = 1 - jaccard_sim
        
        return jaccard_dist


# ==================== Example Usage ====================

def example_usage():
    """Example usage of VietnameseSynonymReplacer"""
    
    replacer = VietnameseSynonymReplacer(
        replace_ratio=0.1,
        tone_aware=True,
        preserve_compounds=True
    )
    
    # Example texts from UIT-VSFC
    texts = [
        "Giảng viên dạy rất tốt và bài giảng dễ hiểu",
        "Tôi rất thích môn học này, nội dung tuyệt vời",
        "Bài giảng khó và giảng viên dạy không rõ ràng"
    ]
    
    print("=" * 60)
    print("SYNONYM REPLACEMENT AUGMENTATION")
    print("=" * 60)
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Original: {text}")
        
        augmented = replacer.augment(text, num_augmentations=3)
        for j, aug_text in enumerate(augmented, 1):
            quality = replacer.compute_quality_score(text, aug_text)
            print(f"   Aug {j}: {aug_text}")
            print(f"   Quality: {quality:.3f}")


if __name__ == "__main__":
    example_usage()