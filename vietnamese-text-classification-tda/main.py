import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

# Khởi tạo JVM trước
import init_jvm  # Import file khởi tạo JVM

# Import các module khác sau khi JVM đã sẵn sàng
from utils.data_utils import get_dataloaders
from models.hybrid_model import get_model
import argparse
# Đóng cảnh báo không cần thiết
import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(trues, preds),
        'f1': f1_score(trues, preds, average='weighted'),
        'precision': precision_score(trues, preds, average='weighted'),
        'recall': recall_score(trues, preds, average='weighted'),
        'conf_matrix': confusion_matrix(trues, preds)
    }
    return metrics

def plot_results(losses, accs, conf_matrix, results_dir):
    plt.figure()
    plt.plot(losses)
    plt.title('Training Loss')
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    
    plt.figure()
    plt.plot(accs)
    plt.title('Validation Accuracy')
    plt.savefig(os.path.join(results_dir, 'acc_curve.png'))
    
    plt.figure()
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, 'conf_matrix.png'))

def save_metrics(metrics, results_dir):
    with open(os.path.join(results_dir, 'evaluation.txt'), 'w') as f:
        for k, v in metrics.items():
            if not isinstance(v, np.ndarray):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}:\n{v}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--task', type=str, default='sentiment', choices=['sentiment', 'topic'])
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['hybrid', 'phobert', 'lstm', 'bilstm', 'gru'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--aug_ratio', type=float, default=0.5)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer, num_classes = get_dataloaders(
        args.data_dir, args.batch_size, args.task, args.aug_ratio
    )
    print("Data loaded successfully.")
    
    print("Initializing model...")
    model = get_model(args.model_type, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    print("Model initialized.")
    
    start_epoch = 0
    if args.resume:
        checkpoints = sorted([f for f in os.listdir('checkpoints') if f.startswith('model_epoch_')])
        if checkpoints:
            latest = checkpoints[-1]
            checkpoint = torch.load(os.path.join('checkpoints', latest))
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print(f"Resumed from epoch {start_epoch}")
    
    losses = []
    accs = []
    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            print(f"Epoch {epoch+1}, batch {i+1}/{len(train_loader)}", end='\r')
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        val_metrics = evaluate(model, val_loader, device)
        accs.append(val_metrics['accuracy'])
        
        print(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
        
        checkpoint = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, os.path.join('checkpoints', f'model_epoch_{epoch+1}.pth'))
    
    print("Training completed. Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    save_metrics(test_metrics, 'results')
    plot_results(losses, accs, test_metrics['conf_matrix'], 'results')
    print("Evaluation saved to results/ folder.")