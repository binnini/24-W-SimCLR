import torch
from torchvision import transforms
import numpy as np

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

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

# Transformations to test
transformations = {
    "original": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "crop": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "flip": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "color_jitter": transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "grayscale": transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "translation": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "shearing": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "rotation": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "center_mask": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        CenterErasing(length=224//4), 
    ]),
    "noise_injection": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda img: img + 0.1 * torch.randn_like(img)),
    ]),
    "kernel_filtering": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(9, 11)),
    ]),
    "random_erasing": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),
    ]),
    "rotation30": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "rotation60": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(degrees=60),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "rotation90": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "horizontal_flip": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "vertical_flip": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "color_jitter_brightness": transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(0.8, 1, 1, 0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "color_jitter_contrast": transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(1, 0.8, 1, 0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "color_jitter_saturation": transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(1, 1, 0.8, 0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "color_jitter_hue": transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(1, 1, 1, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "horizontal_shift": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(0, translate=(0.1, 0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    "vertical_shift": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(0, translate=(0, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
}