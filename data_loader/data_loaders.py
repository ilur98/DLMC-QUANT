import random

from torch.utils.data import Subset
from torchvision import datasets, transforms

from base import BaseDataLoader


class Mnist(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, training=True, drop_last=False,
                 validation_split=0.0, num_workers=1):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, drop_last, validation_split, num_workers)


class CIFAR10(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, training=True, drop_last=False,
                 validation_split=0.0, num_workers=1):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, drop_last, validation_split, num_workers)


class CIFAR100(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, training=True, drop_last=False,
                 validation_split=0.0, num_workers=1):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            trsfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, drop_last, validation_split, num_workers)


class ImageNet(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, training=True, drop_last=False,
                 validation_split=0.0, num_workers=1, image_size=224,
                 n_samples=-1, random_sample=False):
        if training:
            trsfm = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            trsfm = transforms.Compose([
                transforms.Resize(int(image_size / 0.875)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        if n_samples != -1:
            if random_sample:
                nums = list(range(len(self.dataset)))
                choices = random.choices(nums, k=n_samples)
            else:
                choices = list(range(n_samples))
            self.dataset = Subset(self.dataset, choices)
        super().__init__(self.dataset, batch_size, shuffle, drop_last, validation_split, num_workers)
