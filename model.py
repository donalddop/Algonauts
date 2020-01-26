import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, datasets
from skimage import feature
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform
from scipy import io, stats
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, LeaveOneOut
from PIL import Image
from itertools import combinations, product
import imageio

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
    em, lm, meg_avg = testSub_meg.test_meg_submission(folder + '/target_meg.mat', name + '_meg.mat')
    evc, itc, fmri_avg = testSub_fmri.test_fmri_submission(folder + '/target_fmri.mat', name + '_fmri.mat')
    return (em, lm, evc, itc, meg_avg, fmri_avg)

def evaluate_scores(name, folder):
    em, lm, meg = testSub_meg.test_meg_submission(folder + '/target_meg.mat', name + '_meg.mat')
    evc, itc, fmri = testSub_fmri.test_fmri_submission(folder + '/target_fmri.mat', name + '_fmri.mat')
    return meg, fmri

def sq(x):
    return squareform(x, force='tovector', checks=False)

def perceptual_model(image_folder, scale, gauss_sigma, threshold):
    num = 0 # For naming output files
    edges = [] # storing edge detected images
    # Load training data
    image_files = glob.iglob(image_folder)
    # Iterate over all images
    for image in image_files:
        im = imageio.imread(image)
        im_resized = resize(im, (scale, scale))
        # print(im_resized.shape)
        imgray = np.dot(im_resized[...,:3], [0.299, 0.587, 0.114])
        edges_rough = feature.canny(imgray, low_threshold=threshold)
        edges_rough = Image.fromarray(edges_rough)
        edges_smooth = gaussian_filter(edges_rough, sigma=gauss_sigma)
        edges.append(edges_smooth)
        num += 1
        # plt.imshow(im)
        # plt.imshow(imgray, cmap='gray')
        # plt.imshow(edges_rough, cmap='gray')
        # plt.imshow(edges_smooth, cmap='gray')
    # plt.show()
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
    trained_gnb = gnb.fit(X_train, y_train)
    # Cross validation
    # print(np.shape(X_train), np.shape(y_train))
    #
    # kf = KFold(n_splits=5)
    # loo = LeaveOneOut()
    # scores = []
    # for train, test in loo.split(X_train):
    # # for train, test in kf.split(X_train):
    #     # print(np.shape(X_train[train]), np.shape(X_train[test]))
    #     y_pred = gnb.fit(X_train[train], y_train[train]).predict(X_train[test])
    #     accuracy = (sum(y_pred == y_train[test])/len(y_pred))*100
    #     scores.append(accuracy)
    # print("Average accuracy: ", sum(scores)/len(scores))
    return gnb

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

def dnn_rdm(dnn_data, layer):
    dnn_files = glob.iglob(dnn_data)
    activations = []
    num_images = 0
    for file in dnn_files:
        image = loadmat(file)
        # print(np.shape(image[layer]))
        activations.append(np.mean(image[layer], axis=0)) # for fc
        # activations.append(np.mean(image[layer], axis=(0,1))) # for maxpool
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

def plot_images(folder, labels, targets, title):
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
    images = list(glob.iglob(folder))
    labels_correct = [i == j for i,j in zip(labels,targets)]
    labels = [subs.get(item,item) for item in labels]
    for i, file in enumerate(images):
        sub = fig.add_subplot(12, 10, i + 1)
        img = plt.imread(file)
        sub.axis('off')
        sub.set_title(labels[i], fontsize=8, y=-.28)
        if not labels_correct[i]:
            sub.title.set_color('red')
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
    train_labels_78 = np.array([4,4,5,4,4,4,4,4,4,4,
                                4,4,4,4,7,1,1,1,1,1,
                                1,1,1,1,1,1,1,1,1,1,
                                1,1,1,1,1,1,1,1,1,1,
                                1,1,1,1,1,1,1,1,1,1,
                                4,1,2,2,2,2,2,2,2,2,
                                2,2,2,3,3,3,3,3,3,3,
                                3,3,3,3,3,3,3,3])
    # Select the datasets
    training_folder_92 = 'Training_Data/92_Image_Set'
    training_folder_118 = 'Training_Data/118_Image_Set'
    test_folder = 'Test_Data'
    # Set the folder for training and test images
    images_92 = 'Training_Data/92_Image_Set/92images/*.jpg'
    images_118 = 'Training_Data/118_Image_Set/118images/*.jpg'
    images_78 = 'Test_Data/78images/*.jpg'
    # Select the training targets
    fmri_target_92 = loadmat(training_folder_92 + '/target_fmri.mat')
    meg_target_92 = loadmat(training_folder_92 + '/target_meg.mat')
    fmri_target_118 = loadmat(training_folder_118 + '/target_fmri.mat')
    meg_target_118 = loadmat(training_folder_118 + '/target_meg.mat')

    ### Perceptual model

    # p_rdm = perceptual_model(images_92, 100, 2, 0.3) # meg 92
    # p_rdm = perceptual_model(images_92, 133, 2, 0.4) # fmri 92
    # p_rdm = perceptual_model(images_92, 100, 1, .12) # early meg 92 aug
    # p_rdm = perceptual_model(images_92, 166, 2, .13) # evc fmri 92 aug
    # p_rdm = perceptual_model(images_78, 100, 2, .3) # meg 78
    # p_rdm = perceptual_model(images_78, 133, 1.3, .4) # fmri 78

    # print('(em, lm, evc, itc, meg_avg, fmri_avg)')
    p_rdm_early = perceptual_model(images_78, 100, 2, .4) # meg 78
    # save_rdm(p_rdm_early, 'p_rdm')
    # print(evaluate_rdm('p_rdm', test_folder))
    p_rdm_evc = perceptual_model(images_78, 100, 1, .4) # fmri 78
    # save_rdm(p_rdm_evc, 'p_rdm')
    # print(evaluate_rdm('p_rdm', test_folder))

    # # print(evaluate_rdm('p_rdm', training_folder_92))
    # plot_rdm(p_rdm)

    # Optimizing parameters
    # em, lm, evc, itc, meg, fmri, params = ([] for i in range(7))
    # # var_range = np.linspace(100,200,11)
    # scale_range = np.linspace(100,200,4)
    # sigma_c_range = np.linspace(0,5,6)
    # sigma_g_range = np.linspace(0,10,11)
    # threshold_range = np.linspace(0,1,6)
    # # for s in scale_range:
    #     # for c in sigma_c_range:
    # for g in sigma_g_range:
    # # for t in threshold_range:
    #     # p_rdm = perceptual_model(images_78, 100, 2, .4) # meg
    #     p_rdm = perceptual_model(images_78, 100, g, .4) # fmri
    #     save_rdm(p_rdm, 'p_rdm')
    #     # scores = evaluate_rdm('p_rdm', training_folder_92)
    #     scores = evaluate_rdm('p_rdm', test_folder)
    #     em.append(scores[0])
    #     lm.append(scores[1])
    #     evc.append(scores[2])
    #     itc.append(scores[3])
    #     meg.append(scores[4])
    #     fmri.append(scores[5])
    #     params.append(g)
    #
    # # plot_scores(sigma_g_range, em, lm,
    # #             'Early-MEG', 'Late-MEG', 'Gaussian Sigma, Threshold = .3, Scale = 100')
    # # plot_scores(sigma_g_range, evc, itc,
    # #             'EVC-fMRI', 'ITC-fMRI', 'Gaussian Sigma, Threshold = .3, Scale = 100')
    # # plt.show()
    # print(params[meg.index(max(meg))], max(meg))
    # print(params[fmri.index(max(fmri))], max(fmri))

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

    # Estimate categorical RDMs for the 92 set using the 92 classifier
    # categorical_rdm_meg(training target, trained classifier, training layer, dnn_data, labels):
    # itc, evc, classifications = categorical_rdm_fmri(fmri_target_92, classifier_92,
    #                                                 train_layer, dnn_feats_92, train_labels_92)
    # early, late, classifications = categorical_rdm_meg(meg_target_92, classifier_92,
    #                                                 train_layer, dnn_feats_92, train_labels_92)
    # Estimate categorical RDMs for the 118 set using the 118 classifier
    # itc, evc, classifications = categorical_rdm_fmri(fmri_target_118, classifier_118,
    #                                                 train_layer, dnn_feats_118, train_labels_118)
    # early, late, classifications = categorical_rdm_meg(meg_target_118, classifier_118,
    #                                                 train_layer, dnn_feats_118, train_labels_118)
    # Estimate categorical RDMs for the 78 set using the 8x8 RDM based on the 92 set
    # itc_rdm, evc_rdm, classifications = categorical_rdm_fmri(fmri_target_92, classifier_92,
    #                                                 test_layer, dnn_feats_78, train_labels_92)
    # early_rdm, late_rdm, classifications = categorical_rdm_meg(meg_target_92, classifier_92,
    #                                                 test_layer, dnn_feats_78, train_labels_92)
    # Estimate categorical RDMs for the 78 set using the 8x8 RDM based on the 118 set
    itc_rdm, evc_rdm, classifications = categorical_rdm_fmri(fmri_target_118, classifier_118,
                                                    test_layer, dnn_feats_78, train_labels_118)
    early_rdm, late_rdm, classifications = categorical_rdm_meg(meg_target_118, classifier_118,
                                                    test_layer, dnn_feats_78, train_labels_118)

    # Evaluate categorical models
    # # target_folder = training_folder_92
    # # target_folder = training_folder_118
    # target_folder = test_folder
    # save_rdm(early_rdm, 'cat_rdm_early')
    # save_rdm(late_rdm, 'cat_rdm_late')
    # save_rdm(itc_rdm, 'cat_rdm_itc')
    # save_rdm(evc_rdm, 'cat_rdm_evc')
    # plot_rdm(itc_rdm)

    # # print('(early, late, evc, itc, meg_avg, fmri_avg)')
    # # print(evaluate_rdm('cat_rdm_early', target_folder))
    # print(evaluate_rdm('cat_rdm_late', target_folder))
    # # print(evaluate_rdm('cat_rdm_evc', target_folder))
    # print(evaluate_rdm('cat_rdm_itc', target_folder))

    # Plotting image sets with labels
    # folder = 'Training_Data/92_Image_Set/92images/*.jpg'
    # plot_images(folder, classifications, '92 image set')
    # folder = 'Training_Data/118_Image_Set/118images/*.jpg'
    # plot_images(folder, train_labels_118, train_labels_118, '118 image set')
    # folder = 'Test_Data/78images/*.jpg'
    # plot_images(folder, classifications, train_labels_78, '78 image set')

    ### Combining models

    # # Create RDM from DNN features
    # rdm_alexnet_conv1 = dnn_rdm('Feature_Extract/feats/92images_feats/vgg/*.mat', 'maxpool5')
    # rdm_vgg_max5 = dnn_rdm('Feature_Extract/feats/78images_feats/vgg/*.mat', 'maxpool5')
    # # save_rdm(rdm_alexnet_conv1, 'dnn_rdm')
    # # rdm_vgg_fc8 = dnn_rdm('Feature_Extract/feats/92images_feats/vgg/*.mat', 'fc8')
    rdm_vgg_fc8 = dnn_rdm('Feature_Extract/feats/78images_feats/vgg/*.mat', 'fc8')
    # rdm_alexnet_fc8 = dnn_rdm('Feature_Extract/feats/78images_feats/alexnet/*.mat', 'fc8')
    # rdm_vgg_fc8 = dnn_rdm('Feature_Extract/feats/78images_feats/vgg/*.mat', 'fc8')
    # save_rdm(rdm_vgg_fc8, 'dnn_rdm')
    # # print(evaluate_rdm('dnn_rdm', training_folder_92))

    # Combined initial estimate RDMs
    c_rdm_early_late = p_rdm_early #
    c_rdm_evc_late = .7 * p_rdm_early + .3 * late_rdm #
    c_rdm_early_itc = .7 * p_rdm_early + .3 * itc_rdm #
    c_rdm_evc_itc = .8 * p_rdm_early + .2 * itc_rdm #

    # # Plot final estimates
    # final_early_late = .9 * c_rdm_early_late + .1 * rdm_vgg_fc8 #
    # final_evc_late = .8 * c_rdm_evc_late + .2 * rdm_vgg_fc8 #
    # final_early_itc = .9 * c_rdm_early_itc + .1 * rdm_vgg_fc8 #
    # final_evc_itc = .8 * c_rdm_evc_itc + .2 * rdm_vgg_fc8 #
    # save_rdm(final_evc_itc, 'dnn_rdm')
    # print(evaluate_rdm('dnn_rdm', test_folder))
    # plot_rdm(final_evc_itc)

    # Combine models
    em, lm, evc, itc, meg, fmri, params = ([] for i in range(7))
    var_range = np.linspace(0,1,11)
    for w in var_range:
        # combined_rdm = (1-w) * p_rdm_evc + w * itc_rdm # slight boost to meg score at 0.8
        # combined_rdm = (1-w) * c_rdm_early_late + w * rdm_vgg_fc8 #
        # combined_rdm = (1-w) * c_rdm_early_itc + w * rdm_vgg_fc8 #
        # combined_rdm = (1-w) * c_rdm_evc_late + w * rdm_vgg_fc8 #
        combined_rdm = (1-w) * c_rdm_evc_itc + w * rdm_vgg_fc8 #
        save_rdm(combined_rdm, 'combined_rdm')
        scores = evaluate_rdm('combined_rdm', test_folder)
        em.append(scores[0])
        lm.append(scores[1])
        evc.append(scores[2])
        itc.append(scores[3])
        meg.append(scores[4])
        fmri.append(scores[5])
        params.append(w)
    plot_scores(var_range, meg, fmri,
                'avg MEG score', 'avg fMRI score', '% of CNN model')
    plt.show()
    print(params[meg.index(max(meg))], max(meg), fmri[meg.index(max(meg))])
    print(params[fmri.index(max(fmri))], max(fmri), meg[fmri.index(max(fmri))])
