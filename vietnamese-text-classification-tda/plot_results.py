import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Cấu hình đường dẫn ===
LOG_DIR = "logs"
RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"

os.makedirs(RESULTS_DIR, exist_ok=True)

# === 1️⃣ Đọc dữ liệu loss & accuracy từ log huấn luyện ===
def parse_training_log(log_path="logs/training.log"):
    losses, accs = [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Loss:" in line and "Val Acc" in line:
                parts = line.strip().split(" - ")
                loss = float(parts[1].split(":")[1])
                acc = float(parts[2].split(":")[1])
                losses.append(loss)
                accs.append(acc)
    return losses, accs


# === 2️⃣ Vẽ biểu đồ Loss và Accuracy ===
def plot_loss_acc(losses, accs, results_dir=RESULTS_DIR):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(accs, label="Validation Accuracy", color="green", marker="s")
    plt.title("Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "acc_curve.png"))
    plt.close()


# === 3️⃣ Vẽ ma trận nhầm lẫn (Confusion Matrix) ===
def plot_conf_matrix(conf_matrix, results_dir=RESULTS_DIR, labels=["Positive", "Negative", "Neutral"]):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix on Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(results_dir, "conf_matrix.png"))
    plt.close()


# === 4️⃣ Tải kết quả test từ evaluation.txt (đã có trong main.py) ===
def load_conf_matrix(results_dir=RESULTS_DIR):
    cm = []
    with open(os.path.join(results_dir, "evaluation.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    matrix_started = False
    for line in lines:
        if "conf_matrix" in line:
            matrix_started = True
            continue
        if matrix_started:
            if line.strip() == "":
                break
            row = [int(x) for x in line.strip().split()]
            cm.append(row)
    return np.array(cm)


# === 5️⃣ Chạy toàn bộ pipeline ===
if __name__ == "__main__":
    losses, accs = parse_training_log()
    print(f"Loaded {len(losses)} epochs from training log.")
    plot_loss_acc(losses, accs)

    conf_matrix = load_conf_matrix()
    print("Loaded confusion matrix:")
    print(conf_matrix)
    plot_conf_matrix(conf_matrix)

    print("✅ All plots saved to results/ folder!")
