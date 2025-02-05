import sys
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv
import pandas as pd

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

def load_model(filepath):
    model = setup_model()
    model.load_state_dict(torch.load(filepath))
    model.eval()  # 평가 모드로 전환
    return model

if __name__ == "__main__":
    # sys.argv를 사용하여 실행 시 입력받기
    if len(sys.argv) < 2:
        print("Usage: python script.py <transformation_name> <transformation_value>")
        sys.exit(1)
    
    transform_name = sys.argv[1]
    modelPath = f'./model/cifar100_model_{transform_name}.pth'
    if not os.path.exists(modelPath):
        print(f"{transform_name} Model file not found")
        sys.exit(1)

    # 모델 로드
    loaded_model = load_model(modelPath)

    # 각 transform에 대해 평가 수행
    _, test_loader = load_data(None, None)
    overall_accuracy, class_accuracy = evaluate_model(loaded_model, test_loader)
    results = {
        "overall_accuracy": overall_accuracy,
        "class_accuracy": class_accuracy
    }

    # 결과를 저장할 파일 경로
    results_filepath = f'./results/dataAug/per_method/cifar100_results_{transform_name}.csv'

    # CSV 파일로 저장
    with open(results_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 헤더 작성
        headers = ['Class Index', 'Class Name', 'Accuracy']
        writer.writerow(headers)
        
        # 클래스 이름과 인덱스는 CIFAR-100 데이터셋에서 가져옴
        class_names = CIFAR100(root='./data', train=False, download=False).classes
        
        # 각 클래스에 대한 정확도 작성
        for i, class_name in enumerate(class_names):
            row = [i, class_name, results['class_accuracy'][i]]
            writer.writerow(row)
        
        # 전체 정확도 작성
        total_accuracy_row = ['100', 'Total_Accuracy', results['overall_accuracy'], 'N/A']
        writer.writerow(total_accuracy_row)

    print(f"Results saved to {results_filepath}")

    # 결과 출력
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Class Accuracy: {results['class_accuracy']}\n")