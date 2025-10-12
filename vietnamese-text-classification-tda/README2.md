# Vietnamese Text Classification with TDA

**Cải Tiến Hiệu Suất Phân Lớp Văn Bản Tiếng Việt Bằng Cách Kết Hợp Kỹ Thuật Tăng Cường Dữ Liệu với Topo và Học Sâu**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 Giới Thiệu

Đây là implementation cho luận văn thạc sĩ về **phân lớp văn bản tiếng Việt** sử dụng:

- **PhoBERT**: Mô hình BERT pre-trained cho tiếng Việt
- **TDA (Topological Data Analysis)**: Phân tích cấu trúc tô-pô từ attention maps
- **Data Augmentation**: Tăng cường dữ liệu (Synonym Replacement, Back Translation, Contextual)

### 🎯 Mục Tiêu

Cải thiện hiệu suất phân lớp văn bản tiếng Việt **+3-4% F1-score** so với PhoBERT baseline bằng cách:
1. Trích xuất đặc trưng tô-pô từ attention maps của PhoBERT
2. Kết hợp TDA features với semantic features
3. Áp dụng data augmentation phù hợp với tiếng Việt

### 📊 Dataset

**UIT-VSFC** (Vietnamese Students' Feedback Corpus):
- ~16,000 câu phản hồi của sinh viên
- 2 tasks: **Sentiment Analysis** (3 classes) và **Topic Classification** (10 classes)
- Domain: Educational feedback

---

## 🚀 Quick Start

### 1. Cài Đặt

```bash
# Clone repository
git clone https://github.com/your-username/vietnamese-text-classification-tda.git
cd vietnamese-text-classification-tda

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chuẩn Bị Dữ Liệu

Đặt UIT-VSFC dataset vào thư mục `data/raw/`:

```
data/raw/
├── train/
│   ├── sents.txt
│   ├── sentiments.txt
│   └── topics.txt
├── dev/
└── test/
```

### 3. Chạy Thử Nghiệm

**E1: PhoBERT Baseline**
```bash
python scripts/run_experiment.py \
    --config configs/experiment_configs/e1_phobert_baseline.yaml \
    --task sentiment \
    --device auto
```

**E4: Proposed Method (PhoBERT + TDA + DA)**
```bash
python scripts/run_experiment.py \
    --config configs/experiment_configs/e4_proposed.yaml \
    --task sentiment \
    --device auto
```

### 4. Resume Training

Nếu training bị gián đoạn:
```bash
python scripts/run_experiment.py \
    --config configs/experiment_configs/e4_proposed.yaml \
    --resume experiments/checkpoints/e4_proposed/e4_proposed_epoch10.pt
```

---

## 📁 Cấu Trúc Project

```
vietnamese-text-classification-tda/
├── configs/                  # Configuration files
│   ├── base_config.yaml
│   └── experiment_configs/
│       ├── e1_phobert_baseline.yaml
│       ├── e2_phobert_da.yaml
│       ├── e3_phobert_tda.yaml
│       └── e4_proposed.yaml
├── src/                      # Source code
│   ├── data/                 # Data processing
│   ├── models/               # Model architectures
│   ├── tda/                  # TDA modules
│   ├── training/             # Training pipeline
│   ├── evaluation/           # Metrics & tests
│   └── utils/                # Utilities
├── scripts/                  # Execution scripts
│   ├── run_experiment.py
│   └── run_ablation.py
├── notebooks/                # Jupyter notebooks
├── experiments/              # Experiment outputs
│   ├── logs/
│   ├── checkpoints/
│   └── results/
└── data/                     # Data directory
```

---

## 🧪 Experiments

### Main Experiments

| ID | Method | Description |
|----|--------|-------------|
| E0 | TF-IDF + SVM | Traditional baseline |
| E1 | PhoBERT | Transformer baseline |
| E2 | PhoBERT + DA | With data augmentation |
| E3 | PhoBERT + TDA | With topological features |
| **E4** | **PhoBERT + TDA + DA** | **Proposed method** |

### Expected Results

| Method | Sentiment F1 | Topic F1 |
|--------|-------------|----------|
| E0 (TF-IDF) | ~75% | ~72% |
| E1 (PhoBERT) | ~87% | ~76% |
| **E4 (Proposed)** | **~91%** | **~80%** |

### Ablation Studies

- **A1**: Layer selection (0-3, 4-7, 8-11, all)
- **A2**: Homology dimensions (H₀, H₁, H₀+H₁)
- **A3**: PI resolution (10×10, 20×20, 30×30)
- **A4**: Fusion method (concat, attention, gated)
- **A5**: DA technique (SR, BT, Contextual)
- **A6**: DA ratio (10%, 30%, 50%)

---

## 💻 Môi Trường

### Windows PC

```bash
# CPU mode
python scripts/run_experiment.py --config ... --device cpu

# GPU mode (if CUDA available)
python scripts/run_experiment.py --config ... --device cuda
```

### Google Colab

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone và chạy
!git clone https://github.com/your-username/vietnamese-text-classification-tda.git
%cd vietnamese-text-classification-tda
!pip install -r requirements.txt

# Run với Colab GPU
!python scripts/run_experiment.py \
    --config configs/experiment_configs/e4_proposed.yaml \
    --device cuda
```

---

## 📊 Xem Kết Quả

### TensorBoard

```bash
tensorboard --logdir experiments/logs/tensorboard
```

### Load Results

```python
import json

with open('experiments/results/e4_proposed/results.json', 'r') as f:
    results = json.load(f)
    print(f"Test F1: {results['test_metrics']['f1_macro']:.4f}")
```

---

## 🔧 Configuration

Chỉnh sửa config trong `configs/base_config.yaml` hoặc experiment-specific configs:

```yaml
# Example: Reduce batch size for limited memory
dataset:
  batch_size: 8  # Default: 16

# Example: Adjust TDA resolution
model:
  tda:
    persistence_image:
      resolution: 10  # Smaller = faster, Default: 20
```

---

## 📝 Logging

Logs được lưu trong `experiments/logs/`:
- **Main log**: `{experiment_name}_{timestamp}.log`
- **Error log**: `errors/{experiment_name}_errors.log`
- **TensorBoard**: `tensorboard/{experiment_name}/`

---

## 🐛 Debug Mode

Test với subset nhỏ:
```bash
python scripts/run_experiment.py \
    --config configs/experiment_configs/e4_proposed.yaml \
    --debug
```

---

## 📚 Tài Liệu Tham Khảo

### Main References

1. **Kushnareva et al. (2021)** - "Artificial Text Detection via Examining the Topology of Attention Maps" [[Paper]](https://aclanthology.org/2021.emnlp-main.50v2.pdf)

2. **Uchendu & Le (2024)** - "Unveiling Topological Structures in Text: A Comprehensive Survey of TDA in NLP" [[Paper]](https://arxiv.org/pdf/2411.10298)

3. **Nguyen & Nguyen (2020)** - "PhoBERT: Pre-trained language models for Vietnamese" [[Paper]](https://aclanthology.org/2020.findings-emnlp.92/)

### Dataset

**UIT-VSFC**: Vietnamese Students' Feedback Corpus [[Link]](https://nlp.uit.edu.vn/datasets/)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Lưu Cang Kim Long**
- MSHV: 2341863008
- Lớp: 23SCT31
- Email: [your-email@example.com]

**Giáo viên hướng dẫn**: TS. Phạm Thế Anh Phú

---

## 🙏 Acknowledgments

- **VinAI Research** for PhoBERT
- **UIT NLP Group** for UIT-VSFC dataset
- **Scikit-TDA** for TDA libraries

---

## 📞 Contact & Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/vietnamese-text-classification-tda/issues)
- 📧 **Email**: [your-email@example.com]
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/vietnamese-text-classification-tda/discussions)

---

**⭐ If you find this project useful, please consider giving it a star!**




🎯 CÁCH SỬ DỤNG
Bước 1: Cài đặt
pip install -r requirements.txt
Bước 2: Đặt dữ liệu UIT-VSFC vào data/raw/
Bước 3: Chạy Baseline
python scripts/run_experiment.py \
    --config configs/experiment_configs/e1_phobert_baseline.yaml \
    --task sentiment
Bước 4: Chạy Proposed Method
python scripts/run_experiment.py \
    --config configs/experiment_configs/e4_proposed.yaml \
    --task sentiment
Bước 5: Resume nếu bị gián đoạn
python scripts/run_experiment.py \
    --config configs/experiment_configs/e4_proposed.yaml \
    --resume experiments/checkpoints/e4_proposed/e4_proposed_epoch10.pt