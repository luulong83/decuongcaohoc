# Project Structure
🎯 GIẢI PHÁP TỔNG THỂ
Kiến trúc 3-tầng

vietnamese-text-classification-tda/
├── 📊 Data Layer (Preprocessing + Augmentation)
├── 🧠 Model Layer (PhoBERT + TDA + Fusion)
└── 🔬 Experiment Layer (Training + Evaluation + Logging)


`Các Components chính`

1. TDA Module (Persistent Homology + Persistence Images)
2. Data Augmentation Module (SR + BT + Contextual)
3. PhoBERT Integration (Attention extraction + Feature fusion)
4. Training Pipeline (Checkpoint + Resume + Logging)
5. Evaluation Suite (Metrics + Visualization + Statistical tests)

```
vietnamese-text-classification-tda/
│
├── README.md                          # Hướng dẫn cài đặt và sử dụng
├── requirements.txt                   # Dependencies
├── setup.py                          # Package setup
├── .gitignore                        
│
├── configs/                          # ⚙️ Configuration files
│   ├── base_config.yaml              # Base configuration
│   ├── experiment_configs/           
│   │   ├── e0_tfidf_svm.yaml        # E0: TF-IDF baseline
│   │   ├── e1_phobert_baseline.yaml # E1: PhoBERT baseline
│   │   ├── e2_phobert_da.yaml       # E2: PhoBERT + DA
│   │   ├── e3_phobert_tda.yaml      # E3: PhoBERT + TDA
│   │   └── e4_proposed.yaml         # E4: Full model
│   └── ablation_configs/
│       ├── a1_layer_selection.yaml
│       ├── a2_homology_dims.yaml
│       ├── a3_vectorization.yaml
│       ├── a4_fusion_method.yaml
│       ├── a5_da_technique.yaml
│       └── a6_da_ratio.yaml
│
├── data/                             # 📊 Data directory
│   ├── raw/                          # Original UIT-VSFC
│   │   ├── train/
│   │   ├── dev/
│   │   └── test/
│   ├── processed/                    # Preprocessed data
│   ├── augmented/                    # Augmented data
│   └── cache/                        # Cached features
│
├── src/                              # 🔧 Source code
│   ├── __init__.py
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py               # Custom Dataset class
│   │   ├── preprocessor.py          # Text preprocessing
│   │   └── augmentation/
│   │       ├── __init__.py
│   │       ├── synonym_replacement.py
│   │       ├── back_translation.py
│   │       └── contextual_aug.py
│   │
│   ├── models/                       # Model definitions
│   │   ├── __init__.py
│   │   ├── phobert_classifier.py    # PhoBERT baseline
│   │   ├── tda_module.py            # TDA feature extraction
│   │   ├── fusion_model.py          # Feature fusion
│   │   └── baselines/
│   │       ├── tfidf_svm.py         # E0: Traditional baseline
│   │       ├── lstm.py
│   │       └── cnn.py
│   │
│   ├── tda/                          # TDA components
│   │   ├── __init__.py
│   │   ├── persistent_homology.py   # PH computation
│   │   ├── persistence_images.py    # PI vectorization
│   │   ├── attention_processor.py   # Attention map processing
│   │   └── utils.py
│   │
│   ├── training/                     # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main trainer
│   │   ├── checkpoint.py            # Checkpoint management
│   │   └── early_stopping.py
│   │
│   ├── evaluation/                   # Evaluation suite
│   │   ├── __init__.py
│   │   ├── metrics.py               # Accuracy, F1, etc.
│   │   ├── statistical_tests.py     # t-test, Cohen's d
│   │   └── visualization.py         # Plots and figures
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── logger.py                # Custom logger
│       ├── config_loader.py         # Config management
│       └── gpu_utils.py             # GPU/CPU detection
│
├── scripts/                          # 🚀 Execution scripts
│   ├── run_experiment.py            # Main experiment runner
│   ├── run_ablation.py              # Ablation study runner
│   ├── preprocess_data.py           # Data preprocessing
│   ├── augment_data.py              # Data augmentation
│   └── evaluate_model.py            # Model evaluation
│
├── notebooks/                        # 📓 Jupyter notebooks
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_tda_exploration.ipynb     # TDA visualization
│   ├── 03_model_analysis.ipynb      # Model interpretation
│   └── 04_results_visualization.ipynb
│
├── tests/                            # 🧪 Unit tests
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_tda_module.py
│   ├── test_augmentation.py
│   └── test_models.py
│
├── experiments/                      # 📈 Experiment outputs
│   ├── logs/                        # Training logs
│   ├── checkpoints/                 # Model checkpoints
│   ├── results/                     # Evaluation results
│   └── visualizations/              # Figures and plots
│
└── docs/                            # 📚 Documentation
    ├── INSTALL.md                   # Installation guide
    ├── USAGE.md                     # Usage guide
    ├── API.md                       # API documentation
    └── EXPERIMENTS.md               # Experiment documentation
```

---

## Key Files Description

### Core Modules

#### `src/models/phobert_classifier.py`
PhoBERT baseline model with:
- Attention map extraction (layers 8-11)
- CLS token feature extraction
- Classification head

#### `src/tda/persistent_homology.py`
TDA computation:
- Vietoris-Rips filtration
- Persistent homology (H₀, H₁)
- Stability computation

#### `src/tda/persistence_images.py`
Vectorization:
- 20×20 grid Persistence Images
- Gaussian smoothing (σ=0.1)
- Normalization

#### `src/data/augmentation/`
Data augmentation:
- **synonym_replacement.py**: Tone-aware Vietnamese synonyms
- **back_translation.py**: Vi→En→Vi with quality check
- **contextual_aug.py**: PhoBERT MLM-based augmentation

#### `src/training/trainer.py`
Training pipeline:
- Checkpoint saving/loading (resume capability)
- Early stopping
- Learning rate scheduling
- Gradient accumulation

#### `src/utils/logger.py`
Comprehensive logging:
- Console output
- File logging (with rotation)
- TensorBoard integration
- Error tracking

### Experiment Scripts

#### `scripts/run_experiment.py`
```bash
# Run E0: TF-IDF baseline
python scripts/run_experiment.py --config configs/experiment_configs/e0_tfidf_svm.yaml

# Run E4: Proposed method
python scripts/run_experiment.py --config configs/experiment_configs/e4_proposed.yaml --gpu 0
```

#### `scripts/run_ablation.py`
```bash
# Run A1: Layer selection ablation
python scripts/run_ablation.py --config configs/ablation_configs/a1_layer_selection.yaml
```

---

## Platform Support

### Windows PC
```bash
# CPU-only mode
python scripts/run_experiment.py --config configs/experiment_configs/e1_phobert_baseline.yaml --device cpu

# With GPU (if available)
python scripts/run_experiment.py --config configs/experiment_configs/e4_proposed.yaml --device cuda
```

### Google Colab
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/your-repo/vietnamese-text-classification-tda.git
%cd vietnamese-text-classification-tda

# Install dependencies
!pip install -r requirements.txt

# Run experiment with Colab GPU
!python scripts/run_experiment.py --config configs/experiment_configs/e4_proposed.yaml --device cuda
```

---

## Resume Training

```bash
# Resume from checkpoint
python scripts/run_experiment.py \
  --config configs/experiment_configs/e4_proposed.yaml \
  --resume experiments/checkpoints/e4_proposed_epoch10.pt
```

Checkpoint includes:
- Model state_dict
- Optimizer state_dict
- Epoch number
- Best validation score
- Training history

---

## Logging System

### Log Structure
```
experiments/logs/
├── e4_proposed_20250212_143052.log  # Main log
├── tensorboard/                     # TensorBoard logs
│   └── e4_proposed/
└── errors/                          # Error logs
    └── e4_proposed_errors.log
```

### Log Content
- Training progress (loss, accuracy per epoch)
- Validation metrics
- TDA computation time
- Data augmentation statistics
- GPU memory usage
- Error stacktraces

---

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Preprocess data**: `python scripts/preprocess_data.py`
3. **Run baseline**: `python scripts/run_experiment.py --config configs/experiment_configs/e1_phobert_baseline.yaml`
4. **Analyze results**: Open `notebooks/04_results_visualization.ipynb`