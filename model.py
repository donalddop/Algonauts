import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, datasets
from skimage import feature
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform
from scipy import io, stats
from sklearn.naive_bayes import GaussianNB
from PIL import Image
from itertools import combinations, product

import testSub_fmri
import testSub_meg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import glob
import random

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
    plt.legend([name_1, name_2], loc='upper right')
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
    em, lm, _ = testSub_meg.test_meg_submission(folder + '/target_meg.mat', name + '_meg.mat')
    evc, itc, _ = testSub_fmri.test_fmri_submission(folder + '/target_fmri.mat', name + '_fmri.mat')
    return em, lm, evc, itc

def evaluate_scores(name, folder):
    em, lm, meg = testSub_meg.test_meg_submission(folder + '/target_meg.mat', name + '_meg.mat')
    evc, itc, fmri = testSub_fmri.test_fmri_submission(folder + '/target_fmri.mat', name + '_fmri.mat')
    return meg, fmri

def sq(x):
    return squareform(x, force='tovector', checks=False)

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
            # im = gaussian_filter(im, sigma=gauss_sigma)
            imgray = np.dot(im[...,:3], [0.299, 0.587, 0.114])
            # Compute the Canny filter
            edges_rough = feature.canny(imgray, canny_sigma, low_threshold=threshold)
            edges_gray = Image.fromarray(edges_rough)
            edges_smooth = gaussian_filter(edges_gray, sigma=gauss_sigma)
            # edges_smooth = edges_rough
            edges.append(edges_smooth)
            num += 1
    # Create the perceptual RDM using overlapping pixel counts
    rdm = np.zeros((num, num))
    for x in range(num):
        for y in range(num):
            if len(edges) > 0:
                rdm[x,y] = np.count_nonzero(edges[x] & edges[y])
            else:
                rdm[x,y] = 0
            if x == y:
                rdm[x,y] = 0
    # Normalize values
    if rdm.max() > 0:
        rdm = rdm / rdm.max()
    return rdm

def train_classifier(dnn_data, layer, labels):
    # Create the classifier
    gnb = GaussianNB()
    # Gather activations per image in an array: This will be our data
    training_files = glob.iglob(dnn_data)
    train_size = len(labels)
    num_features = 1000
    X_train = np.empty((train_size, num_features))
    row = 0
    for file in training_files:
        current = loadmat(file)
        X_train[row] = current[layer]
        row += 1
    # Labels for the image set
    y_train = labels
    y_pred = gnb.fit(X_train, y_train).predict(X_train)
    accuracy = (int(X_train.shape[0])-(y_train != y_pred).sum())/int(X_train.shape[0])
    # print("Accuracy: ", accuracy)
    # Train the classifier
    return gnb.fit(X_train, y_train)


def categorical_rdm_fmri(fmri_data, trained_classifier, layer, dnn_data, labels):
    # Calculate the categorical 8x8 rdm
    # fmri_data = loadmat(folder + '/target_fmri.mat')
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
        classifications.append(int(trained_classifier.predict(current[layer])))
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
    return cat_rdm_itc, cat_rdm_evc, classifications

def categorical_rdm_meg(meg_data, trained_classifier, layer, dnn_data, labels):
    # meg_data = loadmat(folder + '/target_meg.mat')
    # Average activations over all time intervals
    meg_early = np.mean(meg_data['MEG_RDMs_early'], axis=(0,1))
    meg_late = np.mean(meg_data['MEG_RDMs_late'], axis=(0,1))
    meg_early_rdm = np.empty((8,8))
    meg_late_rdm = np.empty((8,8))
    # Loop over all class combinations
    for i in range(8):
        for j in range(8):
            # Take the mean of all images in the same category
            cat_idx_1 = np.where(labels == i)[0]
            cat_idx_2 = np.where(labels == j)[0]
            cat_rdm_coords = list(product(cat_idx_1, cat_idx_2))
            meg_early_rdm[i,j] = np.mean([meg_early[x] for x in cat_rdm_coords])
            meg_late_rdm[i,j] = np.mean([meg_late[x] for x in cat_rdm_coords])
    # Next we classify the test set and assign values from the categorical rdm
    classifications = []
    train_size = 0
    training_files = glob.iglob(dnn_data)
    for file in training_files:
        current = loadmat(file)
        classifications.append(int(trained_classifier.predict(current[layer])))
        train_size += 1
    cat_rdm_early, cat_rdm_late = (np.empty((train_size, train_size)) for i in range(2))
    for i in range(train_size):
        for j in range(train_size):
            cat_1 = classifications[i]
            cat_2 = classifications[j]
            cat_rdm_early[i,j] = meg_early_rdm[cat_1, cat_2]
            cat_rdm_late[i,j] = meg_late_rdm[cat_1, cat_2]
            if i == j:
                cat_rdm_early[i,j] = 0
                cat_rdm_late[i,j] = 0
    return cat_rdm_early, cat_rdm_late, classifications

# def categorical_rdm(train_folder, test_folder, classifier, layer, dnn_data, labels):
#     # Load the meg data
#     meg_training = loadmat(folder + '/target_meg.mat')
#     meg_early = np.mean(meg_training['MEG_RDMs_early'], axis=(0,1))
#     meg_late = np.mean(meg_training['MEG_RDMs_late'], axis=(0,1))
#     # Load the fmri data
#     fmri_training = loadmat(folder + '/target_fmri.mat')
#     fmri_itc = np.mean(fmri_training['IT_RDMs'], axis=0)
#     fmri_evc = np.mean(fmri_training['EVC_RDMs'], axis=0)
#
#     # Classify the test set and assign values from the categorical rdm
#     classifications = []
#     size = 0
#     training_files = glob.iglob(dnn_data)
#     for file in training_files:
#         current = loadmat(file)
#         classifications.append(int(classifier.predict(current[layer])))
#         size += 1
#
#     # Loop over all class combinations
#     meg_early_rdm = np.empty((8,8))
#     meg_late_rdm = np.empty((8,8))
#     fmri_itc_rdm = np.empty((8,8))
#     fmri_evc_rdm = np.empty((8,8))
#     for i in range(8):
#         for j in range(8):
#             # Take the mean of all images in the same category
#             cat_idx_1 = np.where(labels == i)[0]
#             cat_idx_2 = np.where(labels == j)[0]
#             cat_rdm_coords = list(product(cat_idx_1, cat_idx_2))
#             meg_early_rdm[i,j] = np.mean([meg_early[x] for x in cat_rdm_coords])
#             meg_late_rdm[i,j] = np.mean([meg_late[x] for x in cat_rdm_coords])
#             fmri_itc_rdm[i,j] = np.mean([fmri_itc[x] for x in cat_rdm_coords])
#             fmri_evc_rdm[i,j] = np.mean([fmri_evc[x] for x in cat_rdm_coords])
#
#
#     early, late, itc, evc = (np.empty((size, size)) for i in range(4))
#     for i in range(size):
#         for j in range(size):
#             cat_1 = classifications[i]
#             cat_2 = classifications[j]
#             early[i,j] = meg_early_rdm[cat_1, cat_2]
#             late[i,j] = meg_late_rdm[cat_1, cat_2]
#             itc[i,j] = fmri_itc_rdm[cat_1, cat_2]
#             evc[i,j] = fmri_evc_rdm[cat_1, cat_2]
#             if i == j:
#                 early[i,j] = 0
#                 late[i,j] = 0
#                 itc[i,j] = 0
#                 evc[i,j] = 0
#     return early, late, itc, evc, classifications

def dnn_rdm(dnn_data, layer):
    dnn_files = glob.iglob(dnn_data)
    activations = []
    num_images = 0
    for file in dnn_files:
        image = loadmat(file)
        # print(np.shape(image[layer]))
        activations.append(np.mean(image[layer], axis=0))
        # activations.append(np.mean(image[layer], axis=(0,1)))
        # activations.append(image[layer])
        num_images += 1
    rdm = np.empty((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            # Compute the squared spearman correlation coefficient
            rdm[i,j] = stats.spearmanr(activations[i], activations[j], axis=None)[0] ** 2
            # # Compute the cosine similarity
            # layer_i = sq(activations[i])
            # layer_j = sq(activations[j])
            # product = np.dot(layer_i,layer_j)
            # rdm[i,j] = product / (np.linalg.norm(layer_i) * np.linalg.norm(layer_j))
            if i == j:
                rdm[i,j] = 0
    return rdm
    # print(layer.keys())
    # avg_layer = np.mean(layer['conv1'], axis=(0,1))
    # print(np.shape(avg_layer))
    # stats.spearmanr(a, b=None, axis=0)[source]

def plot_images(folder, labels, title):
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
    fig = plt.figure()
    images = glob.iglob(folder)
    images = [x for _,x in sorted(zip(labels,images))]
    labels_sorted = sorted(labels)
    labels_sorted = [subs.get(item,item) for item in labels_sorted]
    for i, file in enumerate(images):
        sub = fig.add_subplot(10, 8, i + 1)
        img = plt.imread(file)
        sub.axis('off')
        sub.set_title(labels_sorted[i], fontsize=8, y=-.28)
        sub.imshow(img)
    plt.suptitle(title)
    plt.subplot_tool()
    plt.show()

if __name__ == "__main__":
    # Training labels:
    train_labels_92 = np.array([0,1,0,2,1,2,2,0,1,2,
                               0,0,3,3,3,3,3,3,3,3,
                               3,3,3,3,4,4,4,4,4,4,
                               4,4,4,4,4,4,5,5,6,6,
                               5,5,6,6,5,5,5,6,7,7,
                               7,1,7,7,7,7,1,7,7,7,
                               1,1,7,7,7,1,1,7,7,7,
                               1,1,1,1,1,1,1,1,1,1,
                               1,1,1,1,1,1,1,1,1,1,
                               1,1])
    train_labels_118 = np.array([7,1,1,1,1,2,1,7,1,5,
                                4,1,1,1,1,4,4,1,4,1,
                                1,1,1,4,1,4,1,4,1,1,
                                1,4,1,1,1,7,4,1,7,1,
                                7,1,1,1,1,4,1,4,1,1,
                                7,4,4,1,1,4,1,1,4,1,
                                1,1,1,1,1,4,1,7,4,1,
                                4,7,1,7,1,4,1,1,1,1,
                                4,1,1,1,1,4,7,1,1,1,
                                1,1,4,1,4,1,4,1,4,1,
                                1,1,7,1,1,1,1,1,1,1,
                                7,1,1,1,1,1,1,1])

    # # Select the datasets here
    training_folder_92 = 'Training_Data/92_Image_Set'
    training_folder_118 = 'Training_Data/118_Image_Set'
    test_folder = 'Test_Data'
    # load the training and test data
    train_dataset_92 = load_data(training_folder_92, 100)
    fmri_target_92 = loadmat(training_folder_92 + '/target_fmri.mat')
    meg_target_92 = loadmat(training_folder_92 + '/target_meg.mat')
    train_dataset_118 = load_data(training_folder_118, 100)
    test_dataset = load_data(test_folder, 100)

    ### Perceptual model

    # dataset = load_data(test_folder, int(100))
    # p_rdm = perceptual_model(train_dataset_92, 2, 1.5, 0.5) # meg 92
    # # p_rdm = perceptual_model(dataset, 1.5, 1, 0.5) # fmri 92
    p_rdm = perceptual_model(test_dataset, 1.4, 1, 1) # meg 78
    # # p_rdm = perceptual_model(dataset, 1.5, 1.5, 1.5) # fmri 78
    save_rdm(p_rdm, 'p_rdm')
    # # print(evaluate_scores('p_rdm', training_folder))
    # print(evaluate_scores('p_rdm', test_folder))
    # # plot_rdm(p_rdm)

    # Optimizing parameters

    # # dataset = load_data(training_folder, int(100))
    # dataset = load_data(test_folder, int(100))
    #
    # meg, fmri = ([] for i in range(2))
    # # var_range = np.linspace(100,200,11)
    # var_range = np.linspace(0,5,11)
    # for t in var_range:
    #     # dataset = load_data(test_folder, int(t))
    #     # p_rdm = perceptual_model(dataset, t, 1, 1) # meg
    #     p_rdm = perceptual_model(dataset, 1.5, 1.5, 1.5) # fmri
    #     save_rdm(p_rdm, 'p_rdm')
    #     # meg_score, fmri_score = evaluate_scores('p_rdm', training_folder)
    #     meg_score, fmri_score = evaluate_scores('p_rdm', test_folder)
    #     meg.append(meg_score)
    #     fmri.append(fmri_score)
    #
    # plot_scores(var_range, meg, fmri,
    #             'meg', 'fmri', 'Threshold')
    # plt.show()

    ### Categorical model

    # Create categorical rdms from training set
    dnn_feats_92 = 'Feature_Extract/feats/92images_feats/vgg/*.mat'
    dnn_feats_118 = 'Feature_Extract/feats/118images_feats/vgg/*.mat'
    train_layer = 'fc8'
    classifier_92 = train_classifier(dnn_feats_92, train_layer, train_labels_92)
    classifier_118 = train_classifier(dnn_feats_118, train_layer, train_labels_118)
    # Create categorical rdms from test set
    dnn_feats_78 = 'Feature_Extract/feats/78images_feats/vgg/*.mat'
    test_layer = 'fc8'

    # Generate catergorical RDMs
    # # categorical_rdm_meg(folder, trained_classifier, layer, dnn_data, labels):
    # cat_rdm_itc, cat_rdm_evc, classifications = categorical_rdm_fmri(training_folder, classifier_92,
    #                                                 train_layer, dnn_training_92, train_labels_92)
    # cat_rdm_early, cat_rdm_late, classifications = categorical_rdm_meg(training_folder, classifier_92,
    #                                                 train_layer, dnn_training_92, train_labels_92)
    # itc, evc, classifications = categorical_rdm_fmri(fmri_target_92, classifier_92,
    #                                                 train_layer, dnn_feats_92, train_labels_92)
    # early, late, classifications = categorical_rdm_meg(meg_target_92, classifier_92,
    #                                                 train_layer, dnn_feats_92, train_labels_92)
    itc, evc, classifications = categorical_rdm_fmri(fmri_target_92, classifier_92,
                                                    test_layer, dnn_feats_78, train_labels_92)
    early, late, classifications = categorical_rdm_meg(meg_target_92, classifier_92,
                                                    test_layer, dnn_feats_78, train_labels_92)

    # # Plotting image sets with labels
    # folder = 'Training_Data/92_Image_Set/92images/*.jpg'
    # plot_images(folder, classifications, '92 image set')
    # folder = 'Training_Data/118_Image_Set/118images/*.jpg'
    # plot_images(folder, train_labels_118, '118 image set')
    # folder = 'Test_Data/78images/*.jpg'
    # plot_images(folder, classifications, '78 image set')

    # Evaluate categorical models
    # target_folder = training_folder_92
    target_folder = test_folder
    save_rdm(itc, 'cat_rdm_itc')
    save_rdm(evc, 'cat_rdm_evc')
    save_rdm(early, 'cat_rdm_early')
    save_rdm(late, 'cat_rdm_late')
    # print(np.shape(early))
    # print(evaluate_scores('cat_rdm_early', target_folder))
    # print(evaluate_scores('cat_rdm_late', target_folder))
    # print(evaluate_scores('cat_rdm_itc', target_folder))
    # print(evaluate_scores('cat_rdm_evc', target_folder))

    ### Combining models

    # Create RDM from DNN features
    # rdm_alexnet_conv1 = dnn_rdm('Feature_Extract/feats/92images_feats/vgg/*.mat', 'maxpool5')
    # save_rdm(rdm_alexnet_conv1, 'dnn_rdm')
    # rdm_vgg_fc8 = dnn_rdm('Feature_Extract/feats/92images_feats/vgg/*.mat', 'fc8')
    rdm_vgg_fc8 = dnn_rdm('Feature_Extract/feats/78images_feats/vgg/*.mat', 'fc8')
    save_rdm(rdm_vgg_fc8, 'dnn_rdm')
    # print(evaluate_rdm('dnn_rdm', training_folder_92))

    # Combine the categorical and perceptual model
    meg, fmri = ([] for i in range(2))
    var_range = np.linspace(0,1,11)
    for w in var_range:
        # combined_rdm = (1-w) * p_rdm + w * rdm_alexnet_conv1 # slight boost to meg score at 0.8
        combined_rdm = (1-w) * p_rdm + w * rdm_vgg_fc8 #
        # combined_rdm = (1-w) * p_rdm + w * rdm_alexnet_conv1
        save_rdm(combined_rdm, 'combined_rdm')
        # meg_score, fmri_score = evaluate_scores('combined_rdm', training_folder_92)
        meg_score, fmri_score = evaluate_scores('combined_rdm', test_folder)
        meg.append(meg_score)
        fmri.append(fmri_score)

    plot_scores(var_range, meg, fmri,
                'meg', 'fmri', 'w % of categorical model')
    plt.show()
