import timm
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common import config_base, config_change_lr, config_change_batch, config_change_all, train_model, transform_base, transform_augmented, transform_test





# Training ViT with different configs
def train_vit_variant(train_transform, config, save_path):
    full_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=100)
    for name, param in model.named_parameters():
        if "blocks.11" in name or "norm" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_model(model, trainloader, valloader, testloader, save_path, config)


if __name__ == '__main__':
    # Example run
    train_vit_variant(transform_base, config_base, "vit_base_config.pth")
    train_vit_variant(transform_base, config_change_lr, "vit_lr_config.pth")
    train_vit_variant(transform_base, config_change_batch, "vit_batch_config.pth")
    train_vit_variant(transform_base, config_change_all, "vit_all_config.pth")
    train_vit_variant(transform_augmented, config_base, "vit_augmented_config.pth")