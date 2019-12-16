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
from itertools import combinations, product

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

#loads the input files if in .mat format
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def perceptual_model(dataset, canny_sigma, gauss_sigma, threshold):
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
            edges_rough = feature.canny(imgray, canny_sigma, threshold)
            edges_gray = Image.fromarray(edges_rough)
            edges_smooth = gaussian_filter(edges_gray, sigma=gauss_sigma)
            edges.append(edges_smooth)
            # Save the generated image
            # filename = 'Edges/' + folder + '/' + str(num) +'.jpg'
            # plt.imsave(filename, edges[num], cmap=cm.gray)

            num += 1
    # print(edges_smooth)
    #
    # print(edges_gray)
    # plt.imshow(edges_gray, cmap=cm.gray)
    # plt.show()
    # # Apply gaussian filter
    # plt.imshow(edges_0)
    # plt.show()

    # Create the perceptual RDM using overlapping pixel counts
    rdm = np.zeros((num, num))
    for x in range(num):
        for y in range(num):
            rdm[x,y] = np.count_nonzero(edges[x] & edges[y])
            if x == y:
                rdm[x,y] = 0
    # Normalize values
    rdm = rdm / rdm.max()
    # rdm1 = np.ones((num, num))
    # rdm = rdm1 - rdm
    # plt.imshow(edges[0])
    # plt.show()
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

def categorical_model(dataset, folder):
    # Create the classifier
    gnb = GaussianNB()
    # Gather activations per image in an array: This will be our data
    training_files = glob.iglob('Feature_Extract/feats/92images_feats/alexnet/*.mat')
    test_files = glob.iglob('Feature_Extract/feats/78images_feats/alexnet/*.mat')
    train_size = 92
    test_size = 78
    num_features = 1000
    X_train = np.empty((train_size, num_features))
    row = 0
    for file in training_files:
        current = loadmat(file)
        X_train[row] = current['fc8']
        row += 1
    # Label the training images:
    subs = {
        0 : 'Hands',
        1 : 'Objects-Scenes',
        2 : 'Humans',
        3 : 'Faces',
        4 : 'Animals',
        5 : 'Animal Faces',
        6 : 'Monkey Faces',
        7 : 'Fruits-Vegetables',
    }
    y_train = np.array([0,1,0,2,1,2,2,0,1,2,
                       0,0,3,3,3,3,3,3,3,3,
                       3,3,3,3,4,4,4,4,4,4,
                       4,4,4,4,4,4,5,5,6,6,
                       5,5,6,6,5,5,5,6,7,7,
                       7,1,7,7,7,7,1,7,7,7,
                       1,1,7,7,7,1,1,7,7,7,
                       1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,
                       1,1])
    # y_train = [ subs.get(item,item) for item in y_train ]
    # Train the classifier
    trained_classifier = gnb.fit(X_train, y_train)
    y_pred = trained_classifier.predict(X_train)
    accuracy = (int(X_train.shape[0])-(y_train != y_pred).sum())/int(X_train.shape[0])
    # print("Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_pred).sum()))
    print("Accuracy: ", accuracy)
    # print(gnb.fit(X_train, y_train).predict(test_image['fc8']))

    # Next we calculate the categorical 8x8 rdm
    fmri_data = loadmat('Training_Data/92_Image_Set/target_fmri.mat')
    fmri_itc = np.mean(fmri_data['IT_RDMs'], axis=0)
    fmri_evc = np.mean(fmri_data['EVC_RDMs'], axis=0)
    fmri_itc_rdm = np.empty((8,8))
    # Loop over all class combinations
    for i in range(8):
        for j in range(8):
            # Select all images in category
            cat_idx_1 = np.where(y_train == i)[0]
            cat_idx_2 = np.where(y_train == j)[0]
            cat_rdm_coords = list(product(cat_idx_1, cat_idx_2))
            fmri_itc_rdm[i,j] = np.mean([fmri_itc[x] for x in cat_rdm_coords])
    # plot_rdm(fmri_itc_rdm)
    # Next we classify the test set and assign values from the categorical rdm
    classifications = []
    training_files = glob.iglob('Feature_Extract/feats/92images_feats/alexnet/*.mat')
    for file in training_files:
        current = loadmat(file)
        classifications.append(trained_classifier.predict(current['fc8']))
    categorical_rdm = np.empty((train_size, train_size))
    for i in range(train_size):
        for j in range(train_size):
            cat_1 = classifications[i]
            cat_2 = classifications[j]
            categorical_rdm[i,j] = fmri_itc_rdm[cat_1, cat_2]
            if i == j:
                categorical_rdm[i,j] = 0
    # plot_rdm(categorical_rdm)
    return categorical_rdm



def plot_rdm(rdm):
    plt.figure()
    plt.pcolor(rdm)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

def load_data(folder, scale=100):
    # Define rescale transform to (default) 100x100 pixels
    data_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root= folder,
                                           transform=data_transform)
    return dataset

if __name__ == "__main__":
    # Select the datasets here
    training_folder = 'Training_Data/92_Image_Set'
    test_folder = 'Test_Data/78images'
    # load the training and test data
    train_dataset = load_data(training_folder, 200)
    # val_dataset = load_data('92_Image_Set')

    # Perceptual model: input:  (training data, canny sigma, gauss sigma, thres)
    # print("Running perceptual model on: ", training_folder)
    # p_edges, p_rdm = perceptual_model(train_dataset, 1, 1, 0)
    # plot_rdm(p_rdm)
    # save_rdm(p_rdm, 'p_rdm_test')
    # print(testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'p_rdm_test_fmri.mat'))
    # print(testSub_meg.test_meg_submission('Training_Data/92_Image_Set/target_meg.mat', 'p_rdm_test_meg.mat'))

    # Find the best threshold value for edge detection
    scores = []
    var_range = np.linspace(0,3,31)
    for t in var_range:
        p_edges, p_rdm = perceptual_model(train_dataset, 1, t, 0)
        # print(np.shape(p_rdm))
        save_rdm(p_rdm, 'p_rdm_test')
        scores.append(testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'p_rdm_test_fmri.mat'))
    print(scores)
    plt.plot(var_range, scores)
    plt.xlabel('gaussian smoothing sigma')
    plt.ylabel('Score %')
    plt.show()


    # Categorical model
    # print("Running categorical model on: ", training_folder)
    # c_rdm = categorical_model(train_dataset, training_folder)
    #
    # # Combine the categorical and perceptual model
    # scores = []
    # for w in np.linspace(0,1,11):
    #     combined_rdm = (1-w) * p_rdm + w * c_rdm
    #     save_rdm(combined_rdm, 'combined_rdm')
    #     scores.append(testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'combined_rdm_fmri.mat'))
    # print(scores)
    # plt.plot(np.linspace(0,1,11), scores)
    # plt.xlabel('W (% of categorical rdm used)')
    # plt.ylabel('Score %')
    # plt.show()

    # Show the classes and indexes
    # print(train_dataset.class_to_idx)

    # Evaluate rdms
    # testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'p_rdm_fmri.mat')
    # testSub_meg.test_meg_submission('Training_Data/92_Image_Set/target_meg.mat', 'p_rdm_meg.mat')

    # testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'Feature_Extract/rdms/92images_rdms/resnet/pearson/fc/submit_fmri.mat')
    # testSub_fmri.test_fmri_submission('Training_Data/92_Image_Set/target_fmri.mat', 'Feature_Extract/rdms/92images_rdms/alexnet/pearson/conv1/submit_fmri.mat')
