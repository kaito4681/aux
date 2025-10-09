import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tqdm import tqdm

import wandb
from models.plain import (
    plain18,
    plain34,
    plain50,
    plain101,
    plain152,
)


def set_seed(seed):
    """再現性のためにseedを固定"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


cifar100_mean = (0.5071, 0.4865, 0.4409)
cifar100_std = (0.2673, 0.2564, 0.2762)


def calculate_gradient_norm(model):
    """モデルのパラメータの勾配ノルムを計算"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def save_gradient_norms(gradient_norms, run_name, seed, save_dir="result"):
    """勾配ノルムをファイルに保存し、グラフを作成"""
    os.makedirs(save_dir, exist_ok=True)

    # ファイル名にrun_nameとseedを含める
    filename_base = f"{run_name}_seed{seed}"

    # JSONファイルとして保存
    json_path = os.path.join(save_dir, f"{filename_base}_gradient_norms.json")
    with open(json_path, "w") as f:
        json.dump(gradient_norms, f, indent=2)

    # グラフを作成して保存
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(gradient_norms) + 1))
    plt.plot(epochs, gradient_norms, marker="o", linewidth=2, markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Gradient Norm", fontsize=12)
    plt.title(f"Gradient Norm over Training\n{run_name} (seed: {seed})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # グラフを保存
    graph_path = os.path.join(save_dir, f"{filename_base}_gradient_norm.png")
    plt.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gradient norms saved to: {json_path}")
    print(f"Gradient norm graph saved to: {graph_path}")

    return json_path, graph_path


def main():
    # 引数
    parser = argparse.ArgumentParser(
        description="plainにauxiliary classifierを追加する実験"
    )
    parser.add_argument(
        "--aux",
        action="store_true",
        help="auxの付けるか",
    )
    parser.add_argument(
        "--bn",
        action="store_true",
        help="Batch Normalizationを使うかどうか",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="skip connectionを使うかどうか",
    )
    parser.add_argument(
        "--check", action="store_true", help="確認のために1エポックだけ実行"
    )
    parser.add_argument(
        "--use-tqdm", action="store_true", help="進捗表示にtqdmを使うかどうか"
    )
    parser.add_argument("--use-wandb", action="store_true", help="wandbを使うかどうか")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="バッチサイズ (default: 128)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["18", "34", "50", "101", "152"],
        default="50",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # ハイパーパラメータ
    batch_size = args.batch_size
    num_epochs = 1 if args.check else 128
    lr = 1e-3
    weight_decay = 1e-2

    # run_nameの設定
    run_name = "plain" + args.model_size
    if args.bn:
        run_name += "-bn"
    if args.skip:
        run_name += "-skip"
    if args.aux:
        run_name += "-aux"

    # wandb初期化
    if args.use_wandb:
        wandb.init(
            project="aux-bn-skip",
            name=run_name,
            config={
                "model_type": "plainnet" + args.model_size,
                "dataset": "cifar100",
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "optimizer": "AdamW",
                "seed": args.seed,
                "aux": args.aux,
            },
        )

    model_dict = {
        "18": plain18,
        "34": plain34,
        "50": plain50,
        "101": plain101,
        "152": plain152,
    }

    model = model_dict[args.model_size](
        use_bn=args.bn,
        use_skip=args.skip,
        use_aux=args.aux,
        num_classes=100,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # transform
    train_transforms = [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(cifar100_mean, cifar100_std),
    ]

    test_transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(cifar100_mean, cifar100_std),
    ]

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(train_transforms),
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(test_transforms),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_wandb:
        wandb.config.update({"device": str(device)}, allow_val_change=True)

    if device.type == "cpu":
        print(
            "Warning: CUDA is not available. Using CPU for training, which may be slow."
        )

    model.to(device)

    # 最高精度を追跡するための変数
    best_accuracy = 0.0
    best_epoch = 0

    # 勾配ノルムを記録するリスト
    gradient_norms = []

    # 保存ディレクトリの作成

    save_dir = "checkpoints/" + run_name

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_aux_loss = 0.0
        train_aux_correct = 0
        epoch_gradient_norms = []

        # tqdmを使うかどうかで分岐
        train_iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1:3d}/{num_epochs:3d} - Train",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:3d}/{total:3d} [{elapsed}<{remaining},{rate_fmt}{postfix}]",
            )
            if args.use_tqdm
            else train_loader
        )

        for images, labels in train_iterator:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            out, aux_out = model(images)
            train_total += labels.size(0)

            if args.aux:
                loss = criterion(out, labels)
                aux_loss = criterion(aux_out, labels)

                loss = loss + 0.3 * aux_loss
                _, predicted = torch.max(out, 1)
                train_correct += (predicted == labels).sum().item()
                _, aux_predicted = torch.max(aux_out, 1)
                train_aux_correct += (aux_predicted == labels).sum().item()
            else:
                loss = criterion(out, labels)
                _, predicted = torch.max(out, 1)
                train_correct += (predicted == labels).sum().item()

            loss.backward()

            # 勾配ノルムを計算
            grad_norm = calculate_gradient_norm(model)
            epoch_gradient_norms.append(grad_norm)

            optimizer.step()

            train_loss += loss.item()
            if args.aux:
                train_aux_loss += aux_loss.item()

            # tqdmを使っている場合、進捗バーに情報を更新
            if args.use_tqdm and isinstance(train_iterator, tqdm):
                current_acc = 100 * train_correct / train_total
                train_iterator.set_postfix(
                    {
                        "Loss": f"{train_loss / len(train_loader):.4f}",
                        "Acc": f"{current_acc:>6.2f}%",
                    }
                )

        # エポックの平均勾配ノルムを記録
        avg_grad_norm = np.mean(epoch_gradient_norms)
        gradient_norms.append(avg_grad_norm)

        # 評価
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_aux_loss = 0.0
        test_aux_correct = 0

        # tqdmを使うかどうかで分岐
        test_iterator = (
            tqdm(
                test_loader,
                desc=f"Epoch {epoch + 1:3d}/{num_epochs:3d} - Test ",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:3d}/{total:3d} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            if args.use_tqdm
            else test_loader
        )

        with torch.no_grad():
            for images, labels in test_iterator:
                images, labels = images.to(device), labels.to(device)
                out, aux_out = model(images)

                test_total += labels.size(0)

                if args.aux:
                    loss = criterion(out, labels)
                    aux_loss = criterion(aux_out, labels)
                    _, predicted = torch.max(out, 1)
                    test_correct += (predicted == labels).sum().item()
                    _, aux_predicted = torch.max(aux_out, 1)
                    test_aux_correct += (aux_predicted == labels).sum().item()
                    test_loss += loss.item()
                    test_aux_loss += aux_loss.item()

                else:
                    loss = criterion(out, labels)
                    _, predicted = torch.max(out, 1)
                    test_correct += (predicted == labels).sum().item()

                    test_loss += loss.item()

                # tqdmを使っている場合、進捗バーに情報を更新
                if args.use_tqdm and isinstance(test_iterator, tqdm):
                    current_acc = 100 * test_correct / test_total
                    test_iterator.set_postfix(
                        {
                            "Loss": f"{test_loss / len(test_loader):.4f}",
                            "Acc": f"{current_acc:>6.2f}%",
                        }
                    )

        # tqdmを使っている場合は改行を追加して表示を整理
        if args.use_tqdm:
            print()

        # ログ出力
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Learning Rate: {current_lr:.6f}")
        print(f"Gradient Norm: {avg_grad_norm:.6f}")

        # wandbログ
        if args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "gradient_norm": avg_grad_norm,
            }

        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        test_acc = 100 * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        print("Train:")
        print(f"  Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print("Test:")
        print(f"  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        print()
        if args.use_wandb:
            log_dict.update(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "test/loss": test_loss,
                    "test/accuracy": test_acc,
                }
            )
        if args.aux:
            train_aux_acc = 100 * train_aux_correct / train_total
            train_aux_loss_avg = train_aux_loss / len(train_loader)
            test_aux_acc = 100 * test_aux_correct / test_total
            test_aux_loss_avg = test_aux_loss / len(test_loader)
            print("Train(aux):")
            print(f"  Train Loss: {train_aux_loss_avg:.4f}, Acc: {train_aux_acc:.2f}%")
            print(f"  Test  Loss: {test_aux_loss_avg:.4f}, Acc: {test_aux_acc:.2f}%")
            if args.use_wandb:
                log_dict.update(
                    {
                        "train/aux_loss": train_aux_loss_avg,
                        "train/aux_accuracy": train_aux_acc,
                        "test/aux_loss": test_aux_loss_avg,
                        "test/aux_accuracy": test_aux_acc,
                    }
                )
        print()

        # wandbにログを送信
        if args.use_wandb:
            log_dict["best_accuracy"] = best_accuracy
            log_dict["best_epoch"] = best_epoch
            wandb.log(log_dict)

        # 最高精度の更新とモデル保存
        # メインモデル（aux使用時は最終層、通常時は出力層）の精度で判定
        current_accuracy = test_acc

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = epoch + 1

            # モデル保存のためのファイル名を生成
            checkpoint_path = os.path.join(save_dir, "best.pth")

            # チェックポイントの保存
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
                "accuracy": current_accuracy,  # 精度を保存
                "config": {
                    "model_type": "plainnet" + args.model_size,
                    "num_classes": 100,
                    "seed": args.seed,
                },
            }

            torch.save(checkpoint, checkpoint_path)
            print(
                f"✓ New best model saved! Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})"
            )
            print(f"  Saved to: {checkpoint_path}")

        print(f"Current best accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")

        print("-" * 20)
        print()

        # スケジューラーのステップを実行
        scheduler.step()

    # 訓練終了時の結果表示
    print("=" * 50)
    print("Training completed!")
    print(f"Best accuracy: {best_accuracy:.2f}% (achieved at epoch {best_epoch})")
    checkpoint_path = os.path.join(save_dir, "best.pth")
    print(f"Best model saved at: {checkpoint_path}")
    print("=" * 50)

    # 勾配ノルムを保存してグラフ化
    if len(gradient_norms) > 0:
        json_path, graph_path = save_gradient_norms(gradient_norms, run_name, args.seed)
        print("=" * 50)

    # wandb終了
    if args.use_wandb:
        wandb.finish()

    # 訓練終了時にメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
