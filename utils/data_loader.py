from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

    val_set = datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader