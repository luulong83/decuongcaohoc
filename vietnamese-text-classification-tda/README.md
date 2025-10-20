project_root/
├── main.py                  # Script chính để train, evaluate, và test
├── data/                    # Thư mục dữ liệu đầu vào
│   ├── sents.txt            # File văn bản gốc
│   ├── sentiments.txt       # File nhãn cảm xúc
│   └── topics.txt           # File nhãn chủ đề
├── utils/                   # Các hàm hỗ trợ
│   ├── data_utils.py        # Tiền xử lý, EDA, load data
│   └── tda_utils.py         # Trích xuất đặc trưng TDA từ attention maps
├── models/                  # Định nghĩa model
│   └── hybrid_model.py      # Model hybrid PhoBERT + TDA
├── logs/                    # Thư mục log (tự tạo khi chạy)
│   └── training.log         # File log huấn luyện
├── checkpoints/             # Thư mục lưu checkpoint (tự tạo khi chạy)
│   └── model_epoch_X.pth    # File checkpoint sau mỗi epoch
└── requirements.txt         # Danh sách thư viện cần install


vietnamese-text-classification-tda/
├── main.py
├── utils/
│   ├── __init__.py  # Để import như package
│   ├── data_utils.py
│   └── tda_utils.py
├── models/
│   ├── __init__.py
│   └── hybrid_model.py
├── data/  # Tải dataset vào đây, ví dụ: data/train/sents.txt, ...
├── logs/  # Tạo tự động
├── checkpoints/  # Tạo tự động
├── results/  # Tạo tự động cho evaluation
├── VnCoreNLP-1.1.1.jar
└── requirements.txt


DataSet:
`https://nlp.uit.edu.vn/datasets/`