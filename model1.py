import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torchvision.datasets as datasets
from skimage import feature
from scipy.ndimage import gaussian_filter
from scipy import io
from sklearn.naive_bayes import GaussianNB
from PIL import Image
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import h5py
import math
import os

# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# def resize_image(image):
#     return transform.resize(image, (new_h, new_w))
#loads the input files if in .mat format
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def perceptual_model(dataset, folder, sigma):
    num = 0 # For naming output files
    edges = [] # storing edge detected images
    # Load training data using dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                num_workers=4, shuffle=True)

    # Iterate over all batches
    for images, labels in loader:
        # Select an image for edge detection
        for j in range(len(images)):
            # Convert image to numpy array
            im = np.transpose(images[j].numpy(), (1, 2, 0))
            # Convert rgb to grayscale for edge detection
            imgray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
            # Compute the Canny filter for two values of sigma
            edges.append(feature.canny(imgray, sigma))#, high_threshold=12)

            # Apply gaussian filter
            # edges = gaussian_filter(edges, sigma=1)

            # Save the generated image
            filename = 'Edges/' + folder + '/' + str(num) +'.jpg'
            plt.imsave(filename, edges[num], cmap=cm.gray)
            num += 1

    # Create the perceptual RDM using overlapping pixel counts
    rdm = np.zeros((num, num))
    for x in range(num):
        for y in range(num):
            rdm[x,y] = np.count_nonzero(edges[x] & edges[y])
            if x == y:
                rdm[x,y] = 0
    # Normalize values
    rdm = rdm / rdm.max()
    return edges, rdm

def categorical_model(dataset, folder):
    gnb = GaussianNB()
    activations = loadmat("vgg_fc8_fmri.mat")
    print(type(activations))

    # y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

def load_data(folder, scale=100):
    # Define rescale transform to (default) 100x100 pixels
    data_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root='Training_Data/' + folder,
                                           transform=data_transform)
    return dataset

if __name__ == "__main__":
    # Select the datasets here
    training_folder = '92_Image_Set_Labeled'
    # load the training and test data
    train_dataset = load_data(training_folder)
    # val_dataset = load_data('92_Image_Set')

    # Perceptual model
    # print("Running perceptual model on: ", train_folder)
    # p_edges, p_rdm = perceptual_model(train_dataset, training_folder, 1)
    # plt.figure()
    # plt.pcolor(p_rdm)
    # plt.colorbar()
    # plt.show()

    # Categorical model
    # Show the classes and indexes
    print(train_dataset.class_to_idx)
    categorical_model(train_dataset, training_folder)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                                # num_workers=4, shuffle=True)

    # Get pretrained model using torchvision.models as models library
    # model = models.densenet161(pretrained=True)

    # Turn off training for their parameters
    # for param in model.parameters():
    #     param.requires_grad = False
