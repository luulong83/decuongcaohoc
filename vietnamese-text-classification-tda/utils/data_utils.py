import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import logging
import numpy as np
from underthesea import word_tokenize  # Tách từ tiếng Việt

logger = logging.getLogger(__name__)

# --- Khởi tạo thông báo ---
print("Note: VnCoreNLP JVM initialization skipped.")
print("Using alternative Vietnamese text processing methods.")
print("For better word segmentation, install: pip install underthesea")
print("✓ Using underthesea for Vietnamese word segmentation")

# ==============================
# 1️⃣ LOAD DỮ LIỆU
# ==============================
def load_data(data_dir, task='sentiment'):
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Read {len(lines)} lines from {file_path}")
            return lines

    splits = ['train', 'dev', 'test']
    data = {}
    for split in splits:
        sents_path = os.path.join(data_dir, split, 'sents.txt')
        labels_path = os.path.join(data_dir, split, f'{task}s.txt')
        if not os.path.exists(sents_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Missing data files in {os.path.join(data_dir, split)}. "
                f"Expected: {sents_path} and {labels_path}"
            )
        sents = read_file(sents_path)
        labels = [int(l) for l in read_file(labels_path)]
        data[split] = list(zip(sents, labels))
    return data

# ==============================
# 2️⃣ DATA AUGMENTATION (Nhanh, an toàn)
# ==============================

def synonym_replacement(sentence, n=0.1):
    """Viết hoa ngẫu nhiên vài từ để giả lập thay đổi nhẹ"""
    words = word_tokenize(sentence, format='text').split()
    num_words = len(words)
    if num_words == 0:
        return sentence
    n_replace = max(1, int(num_words * n))
    for _ in range(n_replace):
        idx = random.randint(0, num_words - 1)
        words[idx] = words[idx].upper() if random.random() > 0.5 else words[idx]
    return ' '.join(words)

def random_insertion(sentence, n=0.1):
    """Chèn ngẫu nhiên vài từ trong câu"""
    words = word_tokenize(sentence, format='text').split()
    num_words = len(words)
    if num_words == 0:
        return sentence
    n_insert = max(1, int(num_words * n))
    for _ in range(n_insert):
        idx = random.randint(0, num_words)
        words.insert(idx, random.choice(words))
    return ' '.join(words)

def random_deletion(sentence, n=0.1):
    """Xóa ngẫu nhiên vài từ trong câu"""
    words = word_tokenize(sentence, format='text').split()
    if len(words) <= 1:
        return sentence
    n_delete = max(1, int(len(words) * n))
    for _ in range(n_delete):
        idx = random.randint(0, len(words) - 1)
        del words[idx]
    return ' '.join(words)

def augment_data(data, aug_ratio=0.5):
    """Tăng cường dữ liệu nhanh không dùng mạng"""
    augmented = []
    for i, (sent, label) in enumerate(data):
        augmented.append((sent, label))
        # In log nhẹ để kiểm tra tiến độ
        if i % 1000 == 0:
            print(f"Augmenting {i}/{len(data)} samples...")
        # Xác suất thêm dữ liệu tăng cường
        if random.random() < aug_ratio:
            aug_func = random.choice([synonym_replacement, random_insertion, random_deletion])
            try:
                aug_sent = aug_func(sent)
                augmented.append((aug_sent, label))
            except Exception as e:
                logger.warning(f"Augment failed for: {sent[:50]}... | {e}")
    print(f"Data augmented: {len(augmented)} samples (original {len(data)})")
    return augmented

# ==============================
# 3️⃣ DATASET & DATALOADER
# ==============================

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(data_dir, batch_size=16, task='sentiment', aug_ratio=0.5):
    """Tạo DataLoader cho train/val/test"""
    data = load_data(data_dir, task)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    train_data = augment_data(data['train'], aug_ratio)
    val_data = data['dev']
    test_data = data['test']

    train_ds = TextDataset(train_data, tokenizer)
    val_ds = TextDataset(val_data, tokenizer)
    test_ds = TextDataset(test_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    num_classes = len(set([l for _, l in data['train']]))
    return train_loader, val_loader, test_loader, tokenizer, num_classes
