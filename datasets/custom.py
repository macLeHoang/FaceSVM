from torchvision.datasets import ImageFolder
from torchvision import transforms as T

def custom_dataset(root, input_shape=(1, 128, 128)):
    normalize = T.Normalize(mean=[0.5], std=[0.5])

    transforms = T.Compose([
            T.Grayscale(),
            T.RandomCrop(input_shape[1:]),
            T.RandomHorizontalFlip(), 
            T.ToTensor(),
            normalize])

    dataset = ImageFolder(
        root,
        transform=transforms
    ) # tesnor, label
    return dataset