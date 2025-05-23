import numpy as np
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from torchvision import datasets
from torch.utils.data import DataLoader
import timm
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import transform_base, transform_augmented
from common import config_base, config_change_batch
from common import device


def train_svm_variant(transform, config, model_name_to_save):
    print(f"\nTraining SVM with config: {config}")

    # Load dataset
    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    # Feature extractor
    feature_extractor = timm.create_model("resnet18", pretrained=True, num_classes=0)
    feature_extractor.eval().to(device)

    def extract_features(dataloader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Extracting features"):
                inputs = inputs.to(device)
                outputs = feature_extractor(inputs).cpu().numpy()
                features.append(outputs)
                labels.append(targets.numpy())
        return np.vstack(features), np.hstack(labels)

    # Extract features
    train_features, train_labels = extract_features(trainloader)
    test_features, test_labels = extract_features(testloader)

    # Train SVM
    clf = make_pipeline(StandardScaler(), SVC(kernel="linear", decision_function_shape='ovr'))
    clf.fit(train_features, train_labels)

    # Predict and report
    predictions = clf.predict(test_features)
    print(classification_report(test_labels, predictions, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))

    # Save results for later analysis
    np.savez(model_name_to_save,
             train_features=train_features,
             train_labels=train_labels,
             test_features=test_features,
             test_labels=test_labels,
             predictions=predictions)


if __name__ == '__main__':
    # List of (transform, config, filename)
    svm_experiments = [
        (transform_base, config_base, "svm_base_config.npz"),
        (transform_base, config_change_batch, "svm_batch_config.npz"),
        (transform_augmented, config_base, "svm_augmented_config.npz"),
    ]

    for transform, config, filename in svm_experiments:
        train_svm_variant(transform, config, filename)
