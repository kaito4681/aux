import argparse
import os
import random

import torch
import torchvision
from torchvision.models import VisionTransformer as Vit
from tqdm import tqdm

import wandb
from models.vit_aux import VisionTransformerAux as VitAux


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


def main():
    # 引数
    parser = argparse.ArgumentParser(
        description="vitにauxiliary classifierを追加する実験"
    )
    parser.add_argument(
        "--aux",
        action="store_true",
        help="auxiliary classifierを追加するかどうか",
    )
    parser.add_argument(
        "--check", action="store_true", help="確認のために1エポックだけ実行"
    )
    parser.add_argument(
        "--use-tqdm", action="store_true", help="進捗表示にtqdmを使うかどうか"
    )
    parser.add_argument("--use-wandb", action="store_true", help="wandbを使うかどうか")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード (default: 42)")

    args = parser.parse_args()

    # seedを固定
    set_seed(args.seed)

    # wandb初期化
    if args.use_wandb:
        run_name = "vit-base-aux-cifar100" if args.aux else "vit-base-cifar100"
        wandb.init(
            project="aux",
            name=run_name,
            config={
                "model": "vit-base-aux" if args.aux else "vit-base",
                "dataset": "cifar100",
                "seed": args.seed,
                "batch_size": 128,
                "learning_rate": 3e-4,
                "weight_decay": 5e-5,
                "eta_min": 1e-6,
                "warmup_epochs": 16,
                "num_epochs": 1 if args.check else 256,
                "image_size": 32,  # CIFAR-100の元サイズ
                "patch_size": 4,  # 8×8=64パッチになる
                "num_layers": 12,
                "num_heads": 12,
                "hidden_dim": 768,
                "mlp_dim": 3072,
                "num_classes": 100,
            },
        )

    # 定義
    model = (
        # vit-baseと同じパラメータ
        VitAux(
            image_size=32,
            patch_size=4,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=100,  # CIFAR-100用
        )
        if args.aux
        else Vit(
            image_size=32,
            patch_size=4,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=100,  # CIFAR-100用
        )
    )

    batch_size = 128
    num_epochs = 1 if args.check else 256
    lr = 3e-4  # 学習率を下げる
    weight_decay = 5e-5
    eta_min = 1e-6
    warmup_epochs = 16

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + CosineAnnealingLRスケジューラー
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=eta_min
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),  # データ拡張追加
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(cifar100_mean, cifar100_std),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(cifar100_mean, cifar100_std),
            ]
        ),
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

    device = torch.device("cuda")
    model.to(device)

    # 最高精度を追跡するための変数
    best_accuracy = 0.0
    best_epoch = 0

    # 保存ディレクトリの作成
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_aux_loss = None
        train_aux_correct = None

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
            outputs = model(images)
            train_total += labels.size(0)

            if args.aux:
                # mainの計算
                main_loss = criterion(outputs[0], labels)
                _, predicted = torch.max(outputs[0], 1)
                train_correct += (predicted == labels).sum().item()

                # auxの計算
                # auxの数だけaux lossとaux_correctの保存場所を作成
                if train_aux_loss is None:
                    train_aux_loss = [0.0] * len(outputs[1])
                if train_aux_correct is None:
                    train_aux_correct = [0] * len(outputs[1])

                for i, aux_output in enumerate(outputs[1]):
                    # aux 損失
                    train_aux_loss[i] += criterion(aux_output, labels).item()
                    # aux 精度
                    _, aux_predicted = torch.max(aux_output, 1)
                    train_aux_correct[i] += (aux_predicted == labels).sum().item()

                # 最終的なloss
                aux_loss_sum = sum(
                    [criterion(aux_output, labels) for aux_output in outputs[1]]
                )
                loss = main_loss + 0.1 * aux_loss_sum / len(
                    outputs[1]
                )

            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()

            loss.backward()
            # 勾配クリッピングを追加
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            # tqdmを使っている場合、進捗バーに情報を更新
            if args.use_tqdm and isinstance(train_iterator, tqdm):
                current_acc = 100 * train_correct / train_total
                train_iterator.set_postfix(
                    {
                        "Loss": f"{train_loss / train_total:.4f}",
                        "Acc": f"{current_acc:>6.2f}%",
                    }
                )

        # 評価
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_aux_loss = None
        test_aux_correct = None

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
                outputs = model(images)

                test_total += labels.size(0)

                if args.aux:
                    # 最終層のlossの計算
                    loss = criterion(outputs[0], labels)
                    test_loss += loss.item()

                    # 最終層の精度計算
                    _, predicted = torch.max(outputs[0], 1)
                    test_correct += (predicted == labels).sum().item()

                    # aux用の計算
                    # auxの数だけaux lossとaux_correctの保存場所を作成
                    if test_aux_loss is None:
                        test_aux_loss = [0.0] * len(outputs[1])
                    if test_aux_correct is None:
                        test_aux_correct = [0] * len(outputs[1])

                    for i, aux_output in enumerate(outputs[1]):
                        # auxの損失計算
                        test_aux_loss[i] += criterion(aux_output, labels).item()

                        # auxの精度計算
                        _, aux_predicted = torch.max(aux_output, 1)
                        test_aux_correct[i] += (aux_predicted == labels).sum().item()

                else:
                    loss = criterion(outputs, labels)

                    # 損失を累積
                    test_loss += loss.item()

                    # 精度計算
                    _, predicted = torch.max(outputs, 1)
                    test_correct += (predicted == labels).sum().item()

                # tqdmを使っている場合、進捗バーに情報を更新
                if args.use_tqdm and isinstance(test_iterator, tqdm):
                    current_acc = 100 * test_correct / test_total
                    test_iterator.set_postfix(
                        {
                            "Loss": f"{test_loss / test_total:.4f}",
                            "Acc": f"{current_acc:>6.2f}%",
                        }
                    )

        # tqdmを使っている場合は改行を追加して表示を整理
        if args.use_tqdm:
            print()

        # ログ出力
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Learning Rate: {current_lr:.6f}")

        # wandbログ
        if args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
            }

        if args.aux:
            # train main
            print("Train:")
            train_main_acc = 100 * train_correct / train_total
            train_main_loss_avg = train_loss / train_total
            print(
                f"  Main    - Loss: {train_main_loss_avg:.4f}, Acc: {train_main_acc:.2f}%"
            )

            if args.use_wandb:
                log_dict.update(
                    {
                        "train/main_loss": train_main_loss_avg,
                        "train/main_accuracy": train_main_acc,
                    }
                )

            if train_aux_loss is not None and train_aux_correct is not None:
                for i, aux_loss in enumerate(train_aux_loss):
                    aux_acc = 100 * train_aux_correct[i] / train_total
                    aux_loss_avg = aux_loss / train_total
                    print(
                        f"  Aux {i + 1:2d}  - Loss: {aux_loss_avg:.4f}, Acc: {aux_acc:.2f}%"
                    )

                    if args.use_wandb:
                        log_dict.update(
                            {
                                f"train/aux_{i + 1}_loss": aux_loss_avg,
                                f"train/aux_{i + 1}_accuracy": aux_acc,
                            }
                        )
            print()

            # test
            print("Test:")
            test_main_acc = 100 * test_correct / test_total
            test_main_loss_avg = test_loss / test_total
            print(
                f"  Main    - Loss: {test_main_loss_avg:.4f}, Acc: {test_main_acc:.2f}%"
            )

            if args.use_wandb:
                log_dict.update(
                    {
                        "test/main_loss": test_main_loss_avg,
                        "test/main_accuracy": test_main_acc,
                    }
                )

            if test_aux_loss is not None and test_aux_correct is not None:
                for i, aux_loss in enumerate(test_aux_loss):
                    aux_acc = 100 * test_aux_correct[i] / test_total
                    aux_loss_avg = aux_loss / test_total
                    print(
                        f"  Aux {i + 1:2d}  - Loss: {aux_loss_avg:.4f}, Acc: {aux_acc:.2f}%"
                    )

                    if args.use_wandb:
                        log_dict.update(
                            {
                                f"test/aux_{i + 1}_loss": aux_loss_avg,
                                f"test/aux_{i + 1}_accuracy": aux_acc,
                            }
                        )
            print()

        else:
            train_acc = 100 * train_correct / train_total
            train_loss_avg = train_loss / train_total
            test_acc = 100 * test_correct / test_total
            test_loss_avg = test_loss / test_total

            print("Train:")
            print(f"  Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%")
            print("Test:")
            print(f"  Loss: {test_loss_avg:.4f}, Acc: {test_acc:.2f}%")
            print()

            if args.use_wandb:
                log_dict.update(
                    {
                        "train/loss": train_loss_avg,
                        "train/accuracy": train_acc,
                        "test/loss": test_loss_avg,
                        "test/accuracy": test_acc,
                    }
                )

        # wandbにログを送信
        if args.use_wandb:
            wandb.log(log_dict)

        # 最高精度の更新とモデル保存
        # メインモデル（aux使用時は最終層、通常時は出力層）の精度で判定
        current_accuracy = test_main_acc if args.aux else test_acc

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_epoch = epoch + 1

            # モデル保存のためのファイル名を生成
            model_name = "vit_base_aux" if args.aux else "vit_base"
            dataset = "cifar100"
            checkpoint_path = os.path.join(save_dir, f"{model_name}_{dataset}_best.pth")

            # チェックポイントの保存
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
                "loss": current_accuracy,  # 精度を保存
                "config": {
                    "model_type": model_name,
                    "image_size": 32,
                    "patch_size": 4,
                    "num_layers": 12,
                    "num_heads": 12,
                    "hidden_dim": 768,
                    "mlp_dim": 3072,
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
    model_name = "vit_aux" if args.aux else "vit_base"
    checkpoint_path = os.path.join(save_dir, f"{model_name}_best.pth")
    print(f"Best model saved at: {checkpoint_path}")
    print("=" * 50)

    # wandb終了
    if args.use_wandb:
        wandb.finish()


# def load_best_model(checkpoint_path, device="cuda"):
#     """
#     保存された最高精度のモデルを読み込む

#     Args:
#         checkpoint_path (str): チェックポイントファイルのパス
#         device (str): デバイス ('cuda' or 'cpu')

#     Returns:
#         tuple: (model, checkpoint_info)
#     """
#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     config = checkpoint["config"]

#     # モデルの種類に応じてインスタンスを作成
#     if config["model_type"] == "vit_aux":
#         model = VitAux(
#             image_size=config["image_size"],
#             patch_size=config["patch_size"],
#             num_layers=config["num_layers"],
#             num_heads=config["num_heads"],
#             hidden_dim=config["hidden_dim"],
#             mlp_dim=config["mlp_dim"],
#             num_classes=config["num_classes"],
#         )
#     else:
#         model = Vit(
#             image_size=config["image_size"],
#             patch_size=config["patch_size"],
#             num_layers=config["num_layers"],
#             num_heads=config["num_heads"],
#             hidden_dim=config["hidden_dim"],
#             mlp_dim=config["mlp_dim"],
#             num_classes=config["num_classes"],
#         )

#     # 保存された重みを読み込み
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)

#     # チェックポイント情報
#     checkpoint_info = {
#         "epoch": checkpoint["epoch"],
#         "best_accuracy": checkpoint["best_accuracy"],
#         "model_type": config["model_type"],
#         "seed": config["seed"],
#     }

#     print(f"Loaded model: {config['model_type']}")
#     print(
#         f"Best accuracy: {checkpoint['best_accuracy']:.2f}% (Epoch {checkpoint['epoch']})"
#     )

#     return model, checkpoint_info


if __name__ == "__main__":
    main()
