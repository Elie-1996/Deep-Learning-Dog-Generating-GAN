import os
import torch
import torch.tensor
import PreProcessing
from training import train_nn
from NeuralNetwork import Generator, Discriminator, weights_init


def main():
    print("I will generate dogs in the future :)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # initialize neural networks:
    generator = Generator().to(device)
    generator.apply(weights_init)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)

    training_loader = PreProcessing.load()
    train_nn(discriminator, generator, training_loader, device)


if __name__ == '__main__':
    main()
