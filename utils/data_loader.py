from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
    train_loader = DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 2)

    val_set = datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)
    val_loader = DataLoader(val_set, batch_size = 4, shuffle = False, num_workers = 2)

    return train_loader, val_loader