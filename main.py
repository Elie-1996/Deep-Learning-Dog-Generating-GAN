import os
import torch
import torch.tensor
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL.Image import BICUBIC
from math import floor
import random

import numpy as np

import matplotlib.pyplot as plt
from time import time

PC_PATH = ""  # PC path
DRIVE_PATH = "/content/drive/My Drive/DoggosGAN/"  # Colab path

PATH = PC_PATH
# PATH = DRIVE_PATH

IMAGE_PATH = PATH + "data/Images/"
RESULTS_PATH = PATH + "results/"
SAVE_PATH = PATH + "save/state.pth"
ANNOTATION_PATH = PATH + "Annotation/"

LOAD_NETWORK_PARAMETERS = False

BATCH_SIZE = 32
EPOCH = 1000
D_LR = 0.0001
G_LR = 0.0004
BETA1 = 0.5
BETA2 = 0.999
CRITERION = nn.BCELoss()

# This is supposed to help training by setting a stochastic range around 0.9, instead of 1.0
TRAINING_MAX_LABEL = 0.95
TRAINING_MIN_LABEL = 0.85

# we should choose CLASSES to be a number where 120/(CLASSES-1) is divisible.
TOTAL_CLASSES = 120  # 120 = 2*2*2*3*5
CLASSES = 11


def get_data():
    # Wanted 64X64 images
    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.RandomRotation(10.0, resample=BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = datasets.ImageFolder(IMAGE_PATH, transform=transform)

    return data


def load_shuffle_data(data):
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)

    return loader


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),  # batch normalization
            nn.ReLU(True),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, data):
        return self.main(data)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # replace BatchNorm with LayerNorm normalization
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(p=0.5, inplace=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, CLASSES, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data):
        return torch.reshape(self.main(data), (data.size()[0], CLASSES))


def weights_init(m):
    """
    Takes as input a neural network m that will initialize all its weights.
    """
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    print("I will generate dogs in the future :)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # torch.cuda.get_device_name()

    # initialize neural networks:
    generator = Generator().to(device)
    generator.apply(weights_init)
    optimizer_g = optim.Adam(generator.parameters(), lr=G_LR, betas=(BETA1, BETA2))

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LR, betas=(BETA1, BETA2))

    if LOAD_NETWORK_PARAMETERS:
        print("I am using loaded networks.")
        if os.path.exists(SAVE_PATH):
            checkpoint = torch.load(SAVE_PATH)
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

            generator.load_state_dict(checkpoint['generator_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    else:
        print("I am using fresh networks.")

    # Load training data
    image_data = get_data()

    print("---------------------")
    print("Starting training:")
    for epoch in range(EPOCH):

        # shuffle data with each epoch
        training_loader = load_shuffle_data(image_data)
        for i, data in enumerate(training_loader):
            real_image_, real_class_ = data
            real_image = real_image_.clone().detach().to(device)
            real_class = real_class_.clone().detach().to(device)

            # 1st Step: Updating the weights of the neural network of the discriminator
            discriminator.zero_grad()

            # Training the discriminator with a real image of the dataset
            target = torch.zeros(real_image.size()[0], CLASSES, device=device)
            for j, label in enumerate(real_class):
                target[j][floor(label/(TOTAL_CLASSES/(CLASSES-1)))] = random.random()*(TRAINING_MAX_LABEL - TRAINING_MIN_LABEL) + TRAINING_MIN_LABEL  # values [TRAINING_MIN_LABEL, TRAINING_MAX_LABEL] ~ close to 1.0

            output = discriminator(real_image)
            err_d_real = CRITERION(output, target)

            # Training the discriminator with a fake image generated by the generator
            noise = torch.randn(real_image.size()[0], 100, 1, 1, device=device)
            fake = generator(noise)  # already in GPU
            target = torch.zeros(real_image.size()[0], CLASSES, device=device)
            for j in range(real_image.size()[0]):
                target[j][CLASSES-1] = random.random()*(TRAINING_MAX_LABEL - TRAINING_MIN_LABEL) + TRAINING_MIN_LABEL  # values [TRAINING_MIN_LABEL, TRAINING_MAX_LABEL] ~ close to 1.0
            output = discriminator(fake.detach())
            err_d_fake = CRITERION(output, target)

            # Back-propagation of the total error
            err_d = err_d_real + err_d_fake
            err_d.backward()
            optimizer_d.step()

            # 2nd Step: Updating the weights of the neural network of the generator
            generator.zero_grad()
            target = torch.rand(real_image.size()[0], CLASSES, device=device)*(0.9 - 0.4) + 0.4  # request bad c:=confidence for other dogs (0.4 <= c <= 0.9)
            for j in range(real_image.size()[0]):
                target[j][CLASSES-1] = 0.0  # request that the discriminator is 100% certain that it is a REAL image

            output = discriminator(fake)
            err_g = CRITERION(output, target)
            err_g.backward()
            optimizer_g.step()

            # reshape the array to dimensions of: batch x width pixels x height pixels x RGB (for saving purposes)
            # real_image = real_image.numpy().transpose(0, 2, 3, 1)

            # 3rd Step: Printing the losses and saving the real images
            # and the generated images of the minibatch every 100 steps
            print('[%d/%d][%d/%d] Loss_D: %.4f; Loss_G: %.4f' % (
                epoch, EPOCH, i, len(training_loader), err_d.item(), err_g.item()))
            if i % 100 == 0:
                save_image(real_image, f"{RESULTS_PATH}real_samples.png", normalize=True)
                fake = generator(noise)
                save_image(fake.data, f"{RESULTS_PATH}fake_samples_epoch_{epoch}.png", normalize=True)
        # Save models' states every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                # save discriminator state
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),

                # save generator state
                'generator_state_dict': generator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
            }, SAVE_PATH)


if __name__ == '__main__':
    main()
