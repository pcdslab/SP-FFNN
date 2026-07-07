import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "./results_ffnn"
LOG_CSV = os.path.join(RESULTS_DIR, "training_logs.csv")
CM_CSV = os.path.join(RESULTS_DIR, "valid_confusion_matrix_best_epoch.csv")

# Replace these with your true 16 class names in label order 0..15
CLASS_NAMES = [
    "B cell",
    "CD4 T cell",
    "CD8 T cell",
    "Treg",
    "NK cell",
    "Macrophage",
    "Monocyte",
    "Dendritic cell",
    "Neutrophil",
    "Plasma cell",
    "Tumor cell",
    "Endothelial cell",
    "Fibroblast",
    "Mast cell",
    "HRS cell",
    "Other"
]

def plot_losses(log_csv_path):
    df = pd.read_csv(log_csv_path)

    # assume each row is one epoch
    epochs = np.arange(1, len(df) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, df["train_loss"], label="Training loss", linewidth=2)
    plt.plot(epochs, df["valid_loss"], label="Validation loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "loss_only_figure.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

def plot_loss_accuracy(log_csv):
    df = pd.read_csv(log_csv)
    epochs = np.arange(1, len(df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, df["train_loss"], label="Training loss", linewidth=2)
    axes[0].plot(epochs, df["valid_loss"], label="Validation loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(epochs, df["train_acc"], label="Training accuracy", linewidth=2)
    axes[1].plot(epochs, df["valid_acc"], label="Validation accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "loss_accuracy_figure.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

def plot_confusion_matrix(cm_csv, class_names):
    cm_df = pd.read_csv(cm_csv, index_col=0)

    if cm_df.shape[0] != len(class_names):
        raise ValueError(
            f"Number of class names ({len(class_names)}) does not match confusion matrix size ({cm_df.shape[0]})."
        )

    cm_df.index = class_names
    cm_df.columns = class_names

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix for Validation Set")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confusion_matrix_figure.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    if not os.path.exists(LOG_CSV):
        raise FileNotFoundError(f"Missing file: {LOG_CSV}")
    if not os.path.exists(CM_CSV):
        raise FileNotFoundError(f"Missing file: {CM_CSV}")

    plot_losses(LOG_CSV)
    plot_confusion_matrix(CM_CSV, CLASS_NAMES)