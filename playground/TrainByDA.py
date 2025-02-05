import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from Transformations import transformations

# 디바이스 설정 (GPU 사용 가능하면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(train_transforms=None, test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_dataset = CIFAR100(root='./data', train=True, download=False, transform=train_transforms)
    test_dataset = CIFAR100(root='./data', train=False, download=False, transform=test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    return train_loader, test_loader

def setup_model():
    model = resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)  # CIFAR-100 has 100 classes
    return model.to(device)

def train_model(model, train_loader, criterion, optimizer, num_epochs=60):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 100 == 99:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%")
                running_loss = 0.0

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 100
    class_total = [0] * 100
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    overall_accuracy = 100 * correct / total
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(100)]
    return overall_accuracy, class_accuracy

def main(aug_name=None,train_transforms=None, test_transforms=None):
    print(f"Training model with {aug_name} transformation")
    train_loader, test_loader = load_data(train_transforms, test_transforms)
    model = setup_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1.0e-6)
    train_model(model, train_loader, criterion, optimizer)
    saveFileName = f'./model/cifar100_model_{aug_name}.pth'
    torch.save(model.state_dict(), saveFileName)
    return model

def load_model(filepath):
    model = setup_model()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # 평가 모드로 전환
    return model

class CenterErasing(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        y1 = (h - self.length) // 2
        y2 = (h + self.length) // 2
        x1 = (w - self.length) // 2
        x2 = (w + self.length) // 2

        img[:, y1:y2, x1:x2] = 0
        return img

if __name__ == "__main__":
    # sys.argv를 사용하여 실행 시 입력받기
    if len(sys.argv) < 2:
        print("Usage: python script.py <transformation_name> <transformation_value>")
        sys.exit(1)

    transform_name = sys.argv[1]
    # transformations 사전에서 값 가져오기 (없으면 기본값 사용)
    if transform_name in transformations:
        saveFileName = f'./model/cifar100_model_{transform_name}.pth'
        if (os.path.exists(saveFileName)):
            print("There are already trained model with the given name.")
            sys.exit(1)
        else:
            main(transform_name, transformations[transform_name])
            sys.exit(1)
    else:
        print("There are no transformations with the given name.")
        sys.exit(1)