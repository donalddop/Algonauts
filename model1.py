import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, datasets
from skimage import feature
from scipy.ndimage import gaussian_filter
from scipy import io, stats
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

def load_data(folder, scale=100):
    # Define rescale transform to (default) 100x100 pixels
    data_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root= folder, transform=data_transform)
    return dataset

def plot_scores(var_range, score_1, score_2, name_1, name_2, title):
    plt.figure()
    plt.plot(var_range, score_1)
    plt.plot(var_range, score_2)
    plt.legend([name_1, name_2], loc='upper left')
    plt.xlabel(title)
    plt.ylabel('Score %')

def save_rdm(rdm, filename):
    # Save RDMs in challenge submission format
    rdm_fmri={}
    rdm_meg={}
    rdm_fmri['EVC_RDMs'] = rdm
    rdm_fmri['IT_RDMs'] = rdm
    rdm_meg['MEG_RDMs_late'] = rdm
    rdm_meg['MEG_RDMs_early'] = rdm
    io.savemat(filename + '_fmri.mat',rdm_fmri)
    io.savemat(filename + '_meg.mat',rdm_meg)

def plot_rdm(rdm):
    plt.figure()
    plt.pcolor(rdm)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

def evaluate_rdm(name, folder):
    em, lm = testSub_meg.test_meg_submission(folder + '/target_meg.mat', name + '_meg.mat')
    evc, itc = testSub_fmri.test_fmri_submission(folder + '/target_fmri.mat', name + '_fmri.mat')
    return em, lm, evc, itc

def perceptual_model(dataset, canny_sigma, gauss_sigma, threshold):
    num = 0 # For naming output files
    edges = [] # storing edge detected images
    # Load training data using dataloader
    loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                num_workers=4, shuffle=False)
    # Iterate over all batches
    for images, labels in loader:
        # Select an image for edge detection
        for j in range(len(images)):
            # Convert image to numpy array
            im = np.transpose(images[j].numpy(), (1, 2, 0))
            # Convert rgb to grayscale for edge detection
            imgray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
            # Compute the Canny filter
            edges_rough = feature.canny(imgray, canny_sigma, low_threshold=threshold)
            edges_gray = Image.fromarray(edges_rough)
            edges_smooth = gaussian_filter(edges_gray, sigma=gauss_sigma)
            edges.append(edges_smooth)
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

def train_classifier(dnn_data, layer, labels):
    # Create the classifier
    gnb = GaussianNB()
    # Gather activations per image in an array: This will be our data
    training_files = glob.iglob(dnn_data)
    # test_files = glob.iglob('Feature_Extract/feats/78images_feats/alexnet/*.mat')
    train_size = len(labels)
    # test_size = 78
    num_features = 1000
    X_train = np.empty((train_size, num_features))
    row = 0
    for file in training_files:
        current = loadmat(file)
        X_train[row] = current[layer]
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
    # Labels for the image set
    y_train = labels
    # y_train = [ subs.get(item,item) for item in y_train ]

    # Train the classifier
    return gnb.fit(X_train, y_train)
    # y_pred = trained_classifier.predict(X_train)

    # accuracy = (int(X_train.shape[0])-(y_train != y_pred).sum())/int(X_train.shape[0])
    # print("Accuracy: ", accuracy)


def categorical_rdm_fmri(dataset, folder, trained_classifier, layer, dnn_data, labels):
    # Next we calculate the categorical 8x8 rdm
    fmri_data = loadmat(folder + '/target_fmri.mat')
    fmri_itc = np.mean(fmri_data['IT_RDMs'], axis=0)
    fmri_evc = np.mean(fmri_data['EVC_RDMs'], axis=0)
    fmri_itc_rdm = np.empty((8,8))
    fmri_evc_rdm = np.empty((8,8))
    # Loop over all class combinations
    for i in range(8):
        for j in range(8):
            # Take the mean of all images in the same category
            cat_idx_1 = np.where(labels == i)[0]
            cat_idx_2 = np.where(labels == j)[0]
            cat_rdm_coords = list(product(cat_idx_1, cat_idx_2))
            fmri_itc_rdm[i,j] = np.mean([fmri_itc[x] for x in cat_rdm_coords])
            fmri_evc_rdm[i,j] = np.mean([fmri_evc[x] for x in cat_rdm_coords])
    # Next we classify the test set and assign values from the categorical rdm
    classifications = []
    train_size = 0
    training_files = glob.iglob(dnn_data)
    for file in training_files:
        current = loadmat(file)
        classifications.append(trained_classifier.predict(current[layer]))
        train_size += 1
    cat_rdm_evc, cat_rdm_itc = (np.empty((train_size, train_size)) for i in range(2))
    for i in range(train_size):
        for j in range(train_size):
            cat_1 = classifications[i]
            cat_2 = classifications[j]
            cat_rdm_itc[i,j] = fmri_itc_rdm[cat_1, cat_2]
            cat_rdm_evc[i,j] = fmri_evc_rdm[cat_1, cat_2]
            if i == j:
                cat_rdm_itc[i,j] = 0
                cat_rdm_evc[i,j] = 0
    return cat_rdm_itc, cat_rdm_evc

# def categorical_rdm_meg(dataset, folder, labels):
#     meg_data = loadmat(folder + '/target_meg.mat')
#     # Loop over all time intervals
#     for k in range(20):
#         meg_early = np.mean(meg_data['MEG_RDMs_early'], axis=0)
#         meg_late = np.mean(meg_data['MEG_RDMs_late'], axis=0)
#         meg_early_rdm = np.empty((8,8))
#         meg_late_rdm = np.empty((8,8))
#         # Loop over all class combinations
#         for i in range(8):
#             for j in range(8):
#                 # Take the mean of all images in the same category

def dnn_rdm(dnn_data, layer):
    dnn_files = glob.iglob(dnn_data)
    activations = []
    num_images = 0
    # layer = loadmat('Feature_Extract/feats/78images_feats/alexnet/image_01.mat')
    for file in dnn_files:
        image = loadmat(file)
        activations.append(np.mean(image[layer], axis=(0,1)))
        num_images += 1
    rdm = np.empty((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            # print(stats.spearmanr(activations[i], activations[j], axis=None))
            rdm[i,j] = stats.spearmanr(activations[i], activations[j], axis=None)[0] ** 2
            if i == j:
                rdm[i,j] = 0
    print(rdm)

    # print(layer.keys())
    # avg_layer = np.mean(layer['conv1'], axis=(0,1))
    # print(np.shape(avg_layer))
    # stats.spearmanr(a, b=None, axis=0)[source]

if __name__ == "__main__":
    dnn_rdm('Feature_Extract/feats/78images_feats/alexnet/*.mat', 'conv1')
    # Select the datasets here
    # training_folder = 'Training_Data/92_Image_Set'
    # # Training labels:
    #         # 0 : 'Hands',
    #         # 1 : 'Objects-Scenes',
    #         # 2 : 'Humans',
    #         # 3 : 'Faces',
    #         # 4 : 'Animals',
    #         # 5 : 'Animal Faces',
    #         # 6 : 'Monkey Faces',
    #         # 7 : 'Fruits-Vegetables',
    # train_labels_92 = np.array([0,1,0,2,1,2,2,0,1,2,
    #                            0,0,3,3,3,3,3,3,3,3,
    #                            3,3,3,3,4,4,4,4,4,4,
    #                            4,4,4,4,4,4,5,5,6,6,
    #                            5,5,6,6,5,5,5,6,7,7,
    #                            7,1,7,7,7,7,1,7,7,7,
    #                            1,1,7,7,7,1,1,7,7,7,
    #                            1,1,1,1,1,1,1,1,1,1,
    #                            1,1,1,1,1,1,1,1,1,1,
    #                            1,1])
    # test_folder = 'Test_Data'
    # # load the training and test data
    # train_dataset = load_data(training_folder, 100)
    # test_dataset = load_data(test_folder, 100)
    #
    # # Perceptual model: input:  (training data, canny sigma, gauss sigma, thres)
    # p_edges, p_rdm = perceptual_model(train_dataset, 2, 2, 1)
    # save_rdm(p_rdm, 'p_rdm')

    # print(evaluate_rdm('p_rdm', training_folder))
    # plot_rdm(p_rdm)

    # Find the best threshold value for edge detection
    # early_meg, late_meg, evc_fmri, itc_fmri = ([] for i in range(4))
    #
    # var_range = np.linspace(0,5,11)
    # for t in var_range:
    #     # train_dataset = load_data(training_folder, int(t))
    #     p_edges, p_rdm = perceptual_model(test_dataset, 2, 2, 1)
    #     save_rdm(p_rdm, 'p_rdm')
    #     em, lm, evc, itc = evaluate_rdm('p_rdm', test_folder)
    #     early_meg.append(em)
    #     late_meg.append(lm)
    #     evc_fmri.append(evc)
    #     itc_fmri.append(itc)
    # # print(early_meg, late_meg, evc_fmri, itc_fmri)
    # plot_scores(var_range, early_meg, late_meg,
    #             'early_meg', 'late_meg', 'Canny sigma')
    # plot_scores(var_range, evc_fmri, itc_fmri,
    #             'evc_fmri', 'itc_fmri', 'Canny sigma')
    # plt.show()

    # # Categorical model
    # dnn_train_data = 'Feature_Extract/feats/92images_feats/vgg/*.mat'
    # dnn_test_data = 'Feature_Extract/feats/78images_feats/vgg/*.mat'
    # train_layer = 'fc8'
    # test_layer = train_layer
    # classifier_92 = train_classifier(dnn_train_data, train_layer, train_labels_92)
    # # Create categorical rdms from training set
    # cat_rdm_itc, cat_rdm_evc = categorical_rdm_fmri(train_dataset, training_folder, classifier_92,
    #                                                 train_layer, dnn_train_data, train_labels_92)
    # # Create categorical rdms from test set
    # # cat_rdm_itc, cat_rdm_evc = categorical_rdm_fmri(train_dataset, training_folder, classifier_92,
    # #                                                 test_layer, dnn_test_data, train_labels_92)
    #
    # # Combine the categorical and perceptual model
    # early_meg, late_meg, evc_fmri, itc_fmri = ([] for i in range(4))
    # var_range = np.linspace(0,1,11)
    # for w in var_range:
    #     combined_rdm = (1-w) * p_rdm + w * cat_rdm_itc
    #     save_rdm(combined_rdm, 'combined_rdm')
    #     em, lm, evc, itc = evaluate_rdm('combined_rdm', training_folder)
    #     early_meg.append(em)
    #     late_meg.append(lm)
    #     evc_fmri.append(evc)
    #     itc_fmri.append(itc)
    #
    # plot_scores(var_range, evc_fmri, early_meg,
    #             'evc_fmri', 'early_meg', 'w % of cat model')
    # plot_scores(var_range, itc_fmri, late_meg,
    #             'itc_fmri', 'late_meg', 'w % of cat model')
    # plt.show()
