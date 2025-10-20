import torch
from utils.data_utils import get_dataloaders
import logging

# Cấu hình logging để theo dõi tiến trình
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chạy hàm get_dataloaders
if __name__ == '__main__':
    print("Starting to load data loaders...")
    try:
        train_loader, val_loader, test_loader, tokenizer, num_classes = get_dataloaders(
            data_dir=r'C:\Users\HP\Downloads\data\data',  # Thư mục gốc chứa train/, dev/, test/
            batch_size=16,
            task='sentiment',  # Sửa thành 'sentiment'
            aug_ratio=0.5
        )
        print("Data loaders created successfully:")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        print(f"Number of classes: {num_classes}")
        print(f"Tokenizer type: {type(tokenizer)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error in get_dataloaders: {str(e)}", exc_info=True)