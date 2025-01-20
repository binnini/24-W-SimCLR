import torch
import torchvision
import torchvision.transforms.functional as F
import random
from PIL import ImageFilter
import numpy as np

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, method):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        
        if method == "crop":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "flip":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "color_jitter":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "grayscale":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "translation":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "shearing":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomAffine(degrees=0, shear=10),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "rotation":
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomRotation(degrees=30),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "cutout":    # 안됨
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "noise_injection":   # 안됨
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda img: img + 0.1 * torch.randn_like(img)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "kernel_filtering": 
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif method == "random_erasing":       # 안됨
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomErasing(p=0.5),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            # original code
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
