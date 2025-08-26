import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_confusion_matrix(cm, classes, save_path, title="Confusion matrix"):
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=8)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(save_path, bbox_inches='tight', dpi=140)
    plt.close(fig)

def plot_curves(history, save_dir: Path, title_prefix=""):
    # history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    x = np.arange(1, len(history["train_loss"]) + 1)

    # Loss
    fig1 = plt.figure()
    plt.plot(x, history["train_loss"], label="train")
    plt.plot(x, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}Loss")
    plt.legend()
    fig1.savefig(save_dir / "loss.png", bbox_inches="tight", dpi=140)
    plt.close(fig1)

    # Accuracy
    fig2 = plt.figure()
    plt.plot(x, history["train_acc"], label="train")
    plt.plot(x, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix}Accuracy")
    plt.legend()
    fig2.savefig(save_dir / "accuracy.png", bbox_inches="tight", dpi=140)
    plt.close(fig2)

# ----------------------------
# Models
# ----------------------------

class SimpleCNN(nn.Module):
    """
    A small CNN for MNIST:
    - Conv(1, C1, 3, padding=1) + ReLU -> MaxPool(2)
    - Conv(C1, C2, 3, padding=1) + ReLU -> MaxPool(2)
    - Flatten
    - FC -> hidden -> ReLU -> Dropout
    - FC -> 10
    """
    def __init__(self, c1=32, c2=64, hidden=128, pdrop=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=pdrop)
        # After two 2x2 pools, 28x28 -> 7x7 with channels=c2
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits


class MLP(nn.Module):
    """A simple MLP baseline for MNIST."""
    def __init__(self, hidden1=256, hidden2=128, pdrop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 10)
        self.dropout = nn.Dropout(p=pdrop)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # logits

# ----------------------------
# Data
# ----------------------------

def get_dataloaders(batch_size=128, augment=False, workers=2):
    mean, std = (0.1307,), (0.3081,)
    tfms_train = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:
        tfms_train = [
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    tfms_test = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose(tfms_train))
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose(tfms_test))

    # Split train into train/val
    n_total = len(train_ds)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ----------------------------
# Train / Eval
# ----------------------------

def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        acc_sum += accuracy_from_logits(logits, y)
        n_batches += 1
    return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    all_targets, all_preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()
        acc_sum += accuracy_from_logits(logits, y)
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_targets.append(y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    return loss_sum / max(1, n_batches), acc_sum / max(1, n_batches), all_targets, all_preds

def build_model(name: str, **kwargs) -> nn.Module:
    if name == "cnn":
        return SimpleCNN(c1=kwargs.get("c1", 32),
                         c2=kwargs.get("c2", 64),
                         hidden=kwargs.get("hidden", 128),
                         pdrop=kwargs.get("dropout", 0.25))
    elif name == "mlp":
        return MLP(hidden1=kwargs.get("hidden1", 256),
                   hidden2=kwargs.get("hidden2", 128),
                   pdrop=kwargs.get("dropout", 0.3))
    else:
        raise ValueError("Unknown model name. Use 'cnn' or 'mlp'.")

def build_optimizer(name, params, lr, momentum=0.9, weight_decay=0.0):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer. Use 'sgd' or 'adam'.")

def run_experiment(args, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_name = f"{model_name}_opt-{args.optimizer}_lr-{args.lr}_bs-{args.batch_size}_ep-{args.epochs}"
    outdir = Path("./outputs") / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
                                                            augment=args.augment,
                                                            workers=args.workers)
    if model_name == "cnn":
        model = build_model("cnn", c1=args.c1, c2=args.c2, hidden=args.hidden, dropout=args.dropout)
    else:
        model = build_model("mlp", hidden1=args.hidden1, hidden2=args.hidden2, dropout=args.dropout)

    model = model.to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"[{model_name.upper()}] Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device, criterion)

    # Save artifacts
    (outdir / "artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), outdir / "artifacts" / f"{model_name}_state_dict.pt")
    plot_curves(history, outdir, title_prefix=f"{model_name.upper()} - ")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    save_confusion_matrix(cm, classes=list(range(10)), save_path=outdir / "confusion_matrix.png",
                          title=f"{model_name.upper()} Confusion Matrix")

    # Text report
    report = classification_report(y_true, y_pred, digits=4)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # JSON metrics
    summary = {
        "model": model_name,
        "params": n_params,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "augment": args.augment,
        "val_best_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"\n=== {model_name.upper()} SUMMARY ===")
    print(f"Parameters: {n_params:,}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Outputs saved to: {outdir.as_posix()}")
    return summary, outdir

def run_benchmark(args):
    results = {}
    print("\n>>> Running MLP baseline...")
    mlp_summary, mlp_dir = run_experiment(args, "mlp")
    results["mlp"] = mlp_summary

    print("\n>>> Running CNN model...")
    cnn_summary, cnn_dir = run_experiment(args, "cnn")
    results["cnn"] = cnn_summary

    # Quick comparison plot (bar chart with params vs accuracy)
    labels = ["MLP", "CNN"]
    params = [results["mlp"]["params"], results["cnn"]["params"]]
    accs = [results["mlp"]["test_acc"], results["cnn"]["test_acc"]]

    fig = plt.figure()
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, params, width, label="Params")
    plt.bar(x + width/2, accs, width, label="Test Acc")
    plt.xticks(x, labels)
    plt.title("Benchmark: Parameters vs Test Accuracy")
    plt.legend()
    bench_dir = Path("./outputs") / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(bench_dir / "benchmark_params_acc.png", bbox_inches="tight", dpi=140)
    plt.close(fig)

    # Save combined json
    import json
    with open(bench_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark artifacts saved to: {bench_dir.as_posix()}")
    return results

def parse_args():
    p = argparse.ArgumentParser(description="Lab 3 - CNNs on MNIST")
    p.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--augment", action="store_true", help="Use light data augmentation (rotation).")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    # CNN hyperparams
    p.add_argument("--c1", type=int, default=32, help="First conv filters")
    p.add_argument("--c2", type=int, default=64, help="Second conv filters")
    p.add_argument("--hidden", type=int, default=128, help="Hidden units in FC layer")
    # MLP hyperparams
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)

    p.add_argument("--dropout", type=float, default=0.25, help="Dropout p for CNN FC (MLP uses same arg)")
    p.add_argument("--benchmark", action="store_true", help="Train MLP baseline then CNN and compare.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.benchmark:
        run_benchmark(args)
    else:
        run_experiment(args, args.model)

if __name__ == "__main__":
    main()
