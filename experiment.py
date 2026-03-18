import json
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_transforms(cfg: dict, is_train: bool):
    tfms = []

    if is_train:
        rotation = cfg.get("random_rotation", 0)
        if rotation and rotation > 0:
            tfms.append(transforms.RandomRotation(rotation))

        if cfg.get("horizontal_flip", False):
            tfms.append(transforms.RandomHorizontalFlip())

    tfms.append(transforms.ToTensor())

    if cfg.get("normalize", False):
        mean = cfg.get("mean", [0.5])
        std = cfg.get("std", [0.5])
        tfms.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(tfms)


def get_dataset(dataset_name: str, data_dir: str, train: bool, download: bool, transform):
    dataset_map = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "CIFAR10": datasets.CIFAR10
    }

    if dataset_name not in dataset_map:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Choose from: {list(dataset_map.keys())}"
        )

    dataset_cls = dataset_map[dataset_name]
    return dataset_cls(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )


class FlexibleMLP(nn.Module):
    def __init__(self, input_shape, num_classes: int, hidden_sizes, dropout: float):
        super().__init__()
        c, h, w = input_shape
        input_dim = c * h * w

        layers = [nn.Flatten()]
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        cnn_channels,
        kernel_size: int,
        dropout: float,
        input_size
    ):
        super().__init__()

        layers = []
        in_ch = input_channels

        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size[0], input_size[1])
            flattened_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(cfg: dict, sample_shape):
    model_cfg = cfg["model"]
    model_type = model_cfg["type"].lower()
    num_classes = model_cfg["num_classes"]
    dropout = model_cfg.get("dropout", 0.0)

    if model_type == "mlp":
        return FlexibleMLP(
            input_shape=sample_shape,
            num_classes=num_classes,
            hidden_sizes=model_cfg.get("hidden_sizes", [128, 64]),
            dropout=dropout
        )

    if model_type == "cnn":
        return FlexibleCNN(
            input_channels=sample_shape[0],
            num_classes=num_classes,
            cnn_channels=model_cfg.get("cnn_channels", [32, 64]),
            kernel_size=model_cfg.get("cnn_kernel_size", 3),
            dropout=dropout,
            input_size=(sample_shape[1], sample_shape[2])
        )

    raise ValueError("model.type must be 'mlp' or 'cnn'")


def build_optimizer(model: nn.Module, cfg: dict):
    train_cfg = cfg["training"]
    optimizer_name = train_cfg["optimizer"].lower()
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=weight_decay
        )
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError("training.optimizer must be one of: adam, sgd, adamw")


def build_scheduler(optimizer, cfg: dict):
    sched_cfg = cfg.get("scheduler", {})
    if not sched_cfg.get("enabled", False):
        return None

    sched_type = sched_cfg.get("type", "step").lower()

    if sched_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 1),
            gamma=sched_cfg.get("gamma", 0.1)
        )
    if sched_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("t_max", 10)
        )

    raise ValueError("scheduler.type must be 'step' or 'cosine'")


def build_loss(cfg: dict):
    loss_name = cfg["training"].get("loss", "cross_entropy").lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError("Only 'cross_entropy' is supported right now.")


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += x.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train_one_epoch(model, loader, optimizer, loss_fn, device, log_every, gradient_clip):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        if gradient_clip and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_count += x.size(0)

        if log_every > 0 and batch_idx % log_every == 0:
            print(
                f"  Batch {batch_idx}: "
                f"loss={running_loss / running_count:.4f}, "
                f"acc={running_correct / running_count:.4f}"
            )

    epoch_loss = running_loss / running_count
    epoch_acc = running_correct / running_count
    return epoch_loss, epoch_acc


def main():
    config = load_config("config.json")

    set_seed(config["training"].get("seed", 42))
    device = get_device(config["training"].get("device", "auto"))
    print(f"Using device: {device}")

    dataset_cfg = config["dataset"]
    transform_cfg = config["transforms"]

    train_transform = build_transforms(transform_cfg, is_train=True)
    test_transform = build_transforms(transform_cfg, is_train=False)

    train_dataset = get_dataset(
        dataset_name=dataset_cfg["name"],
        data_dir=dataset_cfg["data_dir"],
        train=True,
        download=dataset_cfg.get("download", True),
        transform=train_transform
    )

    test_dataset = get_dataset(
        dataset_name=dataset_cfg["name"],
        data_dir=dataset_cfg["data_dir"],
        train=False,
        download=dataset_cfg.get("download", True),
        transform=test_transform
    )

    sample_x, _ = train_dataset[0]
    sample_shape = tuple(sample_x.shape)

    model = build_model(config, sample_shape).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    loss_fn = build_loss(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 64),
        shuffle=True,
        num_workers=dataset_cfg.get("num_workers", 2),
        pin_memory=dataset_cfg.get("pin_memory", False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"].get("batch_size", 64),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 2),
        pin_memory=dataset_cfg.get("pin_memory", False)
    )

    epochs = config["training"].get("epochs", 5)
    log_every = config["training"].get("log_every", 100)
    gradient_clip = config["training"].get("gradient_clip", 0.0)

    latest_test_acc = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            log_every=log_every,
            gradient_clip=gradient_clip
        )

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")

        if config.get("evaluation", {}).get("run_test", True):
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            print(f"Test:  loss={test_loss:.4f}, acc={test_acc:.4f}")
            latest_test_acc = test_acc

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"LR after scheduler step: {current_lr:.6f}")

    with open("experiment_results.md", "a", encoding="utf-8") as f:
        f.write(f"\n\nTraining Accuracy: {latest_test_acc}\n\n")

    save_path = config["training"].get("save_path", "model.pth")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to: {save_path}")


if __name__ == "__main__":
    main()