import argparse

import torch
import torchvision
from torchvision.models import VisionTransformer as Vit

from models.vit_aux import VisionTransformerAux as VitAux


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
    args = parser.parse_args()

    # 定義
    model = (
        # vit-baseと同じパラメータ
        VitAux(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=100,  # CIFAR-100用
        )
        if args.aux
        else Vit(
            image_size=224,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=100,  # CIFAR-100用
        )
    )

    batch_size = 32
    num_epochs = 128
    lr = 1e-3
    weight_decay = 1e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
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

    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_aux_loss = None
        train_aux_correct = None

        for images, labels in train_loader:
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
                aux_loss_sum = sum([criterion(aux_output, labels) for aux_output in outputs[1]])
                loss = main_loss + 0.3 * aux_loss_sum / len(outputs[1])

            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 評価
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_aux_loss = None
        test_aux_correct = None

        with torch.no_grad():
            for images, labels in test_loader:
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

        # ログ出力
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        if args.aux:
            # train main
            print(
                f"Train Loss: {train_loss / train_total:.4f}, "
                f"Train Accuracy: {100 * train_correct / train_total:.2f}%"
            )
            print()

            if train_aux_loss is not None and train_aux_correct is not None:
                for i, aux_loss in enumerate(train_aux_loss):
                    print(
                        f"Aux {i + 1} Loss: {aux_loss / train_total:.4f}, "
                        f"Aux {i + 1} Accuracy: {100 * train_aux_correct[i] / train_total:.2f}%"
                    )
                    print()
            print()

            # test
            print(
                f"Test Loss: {test_loss / test_total:.4f}, "
                f"Test Accuracy: {100 * test_correct / test_total:.2f}%"
            )
            print()

            if test_aux_loss is not None and test_aux_correct is not None:
                for i, aux_loss in enumerate(test_aux_loss):
                    print(
                        f" Aux {i + 1} Loss: {aux_loss / test_total:.4f}, "
                        f"Aux {i + 1} Accuracy: {100 * test_aux_correct[i] / test_total:.2f}%"
                    )
                    print()
            print()

        else:
            print(
                f"Train Loss: {train_loss / train_total:.4f}, "
                f"Train Accuracy: {100 * train_correct / train_total:.2f}%"
            )
            print()
            print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
                  f"Test Accuracy: {100 * test_correct / test_total:.2f}%")
            print()
        
        print("-" * 20)
        print()


if __name__ == "__main__":
    main()
