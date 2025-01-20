import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Basic Model Training")
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use')
    parser.add_argument('--dataset_dir', type=str, default='./datasets', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    return parser.parse_args()

def train(args, loader, model, criterion, optimizer):
    model.train()
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        x = x.to(args.device)
        y = y.to(args.device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        loss_epoch += loss.item()
    return loss_epoch / len(loader), accuracy_epoch / len(loader)

def test(args, loader, model, criterion):
    model.eval()
    loss_epoch = 0
    accuracy_epoch = 0
    class_correct = {i: 0 for i in range(args.num_classes)}
    class_total = {i: 0 for i in range(args.num_classes)}
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            loss = criterion(output, y)
            predicted = output.argmax(1)
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            loss_epoch += loss.item()
            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    class_accuracy = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(args.num_classes)}
    for class_id, accuracy in class_accuracy.items():
        print(f"Class {class_id}: Accuracy = {accuracy:.2f}")
    return loss_epoch / len(loader), accuracy_epoch / len(loader)

if __name__ == "__main__":
    args = parse_args()
    model = resnet18(pretrained=False, num_classes=args.num_classes)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    if args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        test_loss, test_acc = test(args, test_loader, model, criterion)
        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")