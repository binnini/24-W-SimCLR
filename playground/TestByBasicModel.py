import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
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

# 결과를 저장할 딕셔너리
results = {}

# 모델 로드
loaded_model = load_model('./model/cifar100_model_original.pth')

# 각 transform에 대해 평가 수행
for name, transform in transformations.items():
    _, test_loader = load_data(None, transform)
    overall_accuracy, class_accuracy = evaluate_model(loaded_model, test_loader)
    results[name] = {
        "overall_accuracy": overall_accuracy,
        "class_accuracy": class_accuracy
    }

# CIFAR-100 클래스 이름 가져오기
cifar100_classes = CIFAR100(root='./data', train=False).classes

# 결과를 저장할 파일 경로
results_filepath = './results/basic_model/cifar100_basic_results.csv'

# results 딕셔너리를 CSV 파일로 저장
with open(results_filepath, 'w', newline='') as csvfile:
    fieldnames = ['Class Index', 'Class Name', 'original'] + [key for key in transformations.keys() if key != 'original']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for class_index in range(100):
        row = {'Class Index': class_index, 'Class Name': cifar100_classes[class_index], 'original': results['original']['class_accuracy'][class_index]}
        for name in transformations.keys():
            row[name] = results[name]['class_accuracy'][class_index]
        writer.writerow(row)

    # Total Accuracy 행 추가
    total_row = {'Class Index': 100, 'Class Name': 'Total_Accuracy', 'original': results['original']['overall_accuracy']}
    for name in transformations.keys():
        total_row[name] = results[name]['overall_accuracy']
    writer.writerow(total_row)

print(f"Results saved to {results_filepath}")

# 결과 출력
for name, result in results.items():
    print(f"Transform: {name}")
    print(f"Overall Accuracy: {result['overall_accuracy']:.2f}%")
    print(f"Class Accuracy: {result['class_accuracy']}\n")