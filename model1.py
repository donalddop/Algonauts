import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import torchvision.datasets as datasets
from skimage import feature
from scipy.ndimage import gaussian_filter
from scipy import io
from sklearn.naive_bayes import GaussianNB

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

def edge_detect(train_set, train_iter, sigma):
    num = 0 # For naming output files

    # Iterate over all batches
    for images, labels in train_iter:
        # Select an image for edge detection
        for j in range(len(images)):
            # Convert image to numpy array
            im = np.transpose(images[j].numpy(), (1, 2, 0))

            # Convert rgb to grayscale for edge detection
            imgray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
            # Compute the Canny filter for two values of sigma
            edges = feature.canny(imgray, sigma)#, high_threshold=12)

            # Apply gaussian filter
            # edges = gaussian_filter(edges, sigma=1)

            # Save the generated image
            filename = 'Edges/' + train_set + '/' + str(num) +'.jpg'
            plt.imsave(filename, edges, cmap=cm.gray)
            num += 1
    #edges = gaussian_filter(edges, sigma=.5)
    plt.imshow(edges, cmap=cm.gray)
    plt.show()

def GNB():
    gnb = GaussianNB()
    activations = loadmat("Feature_Extract/rdms/92images_rdms/vgg/pearson/fc8/submit_fmri.mat")
    print(type(activations))

    # y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)

def load_data(dataset_name, scale=100):
    # Define rescale transform to (default) 100x100 pixels
    data_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root='Training_Data/' + dataset_name,
                                           transform=data_transform)

    return dataset, data_loader



if __name__ == "__main__":
    # load the training and test data
    train_dataset = load_data('92_Image_Set_Labeled')
    val_dataset = load_data('92_Image_Set')
    # Show the classes and indexes
    print(train_dataset.class_to_idx)
    # print(train_dataset.imgs)

    # Load training data using dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                num_workers=4, shuffle=True))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                                num_workers=4, shuffle=True))

    # Get pretrained model using torchvision.models as models library
    model = models.densenet161(pretrained=True)
    # Turn off training for their parameters
    for param in model.parameters():
        param.requires_grad = False
    # Edge detection
    # print("Running edge detection on: ", train_dataset)
    # edge_detect(train_dataset, iter(train_loader), 1)

    # GNB()
