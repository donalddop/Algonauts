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

import testSub_fmri
import testSub_meg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

import argparse
import zipfile
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

def save_rdm(rdm, filename):
    #saving RDMs in challenge submission format
    rdm_fmri={}
    rdm_meg={}
    rdm_fmri['EVC_RDMs'] = rdm
    rdm_fmri['IT_RDMs'] = rdm
    rdm_meg['MEG_RDMs_late'] = rdm
    rdm_meg['MEG_RDMs_early'] = rdm
    io.savemat(filename + '_fmri.mat',rdm_fmri)
    io.savemat(filename + '_meg.mat',rdm_meg)

    # #creating zipped file for submission
    # zipfmri = zipfile.ZipFile(RDM_filename_fmri_zip, 'w')
    # zipmeg = zipfile.ZipFile(RDM_filename_meg_zip, 'w')
    # os.chdir(RDM_dir)
    # zipfmri.write('submit_fmri.mat')
    # zipmeg.write('submit_meg.mat')
    # zipfmri.close()
    # zipmeg.close()

def categorical_model(dataset, folder):
    # Create the classifier
    gnb = GaussianNB()
    # Gather activations per image in an array: This will be our data
    training_files = glob.iglob('Feature_Extract/feats/92images_feats/alexnet/*.mat')
    test_image = loadmat('Feature_Extract/feats/78images_feats/alexnet/image_01.mat')
    num_images = 92
    num_features = 1000
    X_test = np.empty((num_images, num_features))
    row = 0
    for file in training_files:
        current = loadmat(file)
        X_test[row] = current['fc8']
        row += 1
    # Label the training images
    subs = {
        1 : 'Hands',
        2 : 'Objects-Scenes',
        3 : 'Humans',
        4 : 'Faces',
        5 : 'Animals',
        6 : 'Animal Faces',
        7 : 'Monkey Faces',
        8 : 'Fruits-Vegetables',
    }
    y_test = np.array([1,2,1,3,2,3,3,1,2,3,
                       1,1,4,4,4,4,4,4,4,4,
                       4,4,4,4,5,5,5,5,5,5,
                       5,5,5,5,5,5,6,6,7,7,
                       6,6,7,7,6,6,6,7,8,8,
                       8,2,8,8,8,8,2,8,8,8,
                       2,2,8,8,8,2,2,8,8,8,
                       2,2,2,2,2,2,2,2,2,2,
                       2,2,2,2,2,2,2,2,2,2,
                       2,2])
    y_test = [ subs.get(item,item) for item in y_test ]
    # Train the classifier
    y_pred = gnb.fit(X_test, y_test).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    print(gnb.fit(X_test, y_test).predict(test_image['fc8']))

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
    # save_rdm(p_rdm, 'p_rdm')
    # plt.figure()
    # plt.pcolor(p_rdm)
    # plt.colorbar()
    # plt.show()

    # Categorical model
    # Show the classes and indexes
    # print(train_dataset.class_to_idx)
    categorical_model(train_dataset, training_folder)

    # Evaluate rdms
    # testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'p_rdm_fmri.mat')
    # testSub_meg.test_meg_submission('Training_Data/92_Image_Set/target_meg.mat', 'p_rdm_meg.mat')

    # testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'Feature_Extract/rdms/92images_rdms/resnet/pearson/fc/submit_fmri.mat')
    # testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv1/submit_fmri.mat')
