import torch
import torchvision

from models.vision_transformer import VisionTransformer


def main():
    # vit_config = dict(
    #     image_size=224,
    #     patch_size=32,
    #     num_layers=12,
    #     num_heads=12,
    #     hidden_dim=768,
    #     mlp_dim=3072,
    # )

    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )

    batch_size = 32

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
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model.train()
    


if __name__ == "__main__":
    main()
