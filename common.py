import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from torchvision import transforms, datasets

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# === Configuration Setups ===
config_base = {
    'lr': 5e-4,
    'batch_size': 64,
    'optimizer': 'adam',
    'epochs': 10,
    'patience': 5
}

config_change_lr = {
    **config_base,
    'lr': 1e-4,
    'weight_decay': 1e-4,
}

config_change_batch = {
    **config_base,
    'batch_size': 128,
    'weight_decay': 1e-4,
}

config_change_all = {
    **config_base,
    'lr': 1e-4,
    'batch_size': 128,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 1e-4,
}


# Define data augmentation variations
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_augmented = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
# Training loop function
def train_model(model, trainloader, valloader, testloader, save_path, config):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=config['lr'], weight_decay=config.get('weight_decay', 0))
    else: # config['optimizer'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=config['lr'], momentum=config.get('momentum', 0.9),
                              weight_decay=config.get('weight_decay', 0))

    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        total_train, correct_train = 0, 0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1} - Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
        train_acc = 100 * correct_train / total_train

        model.eval()
        total_val, correct_val = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
        val_acc = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"\u2705 New best model saved (Val Acc: {val_acc:.2f}%) to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n\u23F0 Early stopping at epoch {epoch+1}. Best Val Acc: {best_val_acc:.2f}%")
                break

        scheduler.step()

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    total_test, correct_test = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    print(f"Test Accuracy: {100 * correct_test / total_test:.2f}%")
    del model
    gc.collect()
    torch.cuda.empty_cache()
