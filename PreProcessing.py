import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

PATH = "data/"
IMAGE_PATH = PATH + "Images/"
ANNOTATION_PATH = PATH + "Annotation/"
BATCH_SIZE = 32


def load():
    # Wanted 64X64 images
    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = datasets.ImageFolder(IMAGE_PATH, transform=transform)
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)

    imgs, labels = next(iter(loader))
    imgs = imgs.numpy().transpose(0, 2, 3, 1)
    plt.imshow(imgs[0])
    plt.show()
    return imgs, labels
