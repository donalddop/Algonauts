# This script generates RDMs from the activations of a DNN
#Input
#   --feat_dir : directory that contains the activations generated using generate_features.py
#   --save_dir : directory to save the computed RDM
#   --dist : dist used for computing RDM (e.g. 1-Pearson's R)
#   Note: If you have generated activations of your models using your own code, please replace
#      -   "get_layers_ncondns" to get num of layers, layers list and num of images, and
#      -   "get_features" functions to get activations of a particular layer (layer) for a particular image (i).
#Output
#   Model RDM for the representative layers of the DNN.
#   The output RDM is saved in two files submit_MEG.mat and submit_fMRI.mat in a subdirectory named as layer name.

import json
import glob
import os
import numpy as np
import datetime
import scipy.io as sio
import argparse
import zipfile

from tqdm import tqdm
from utils import zip

def get_layers_ncondns(feat_dir):
    """
    to get number of representative layers in the DNN,
    and number of images(conditions).
    Input:
    feat_dir: Directory containing activations generated using generate_features.py
    Output:
    num_layers: number of layers for which activations were generated
    num_condns: number of stimulus images
    PS: This function is specific for activations generated using generate_features.py
    Write your own function in case you use different script to generate features.
    """
    activations = glob.glob(feat_dir + "/*" + ".mat")
    num_condns = len(activations)
    feat=sio.loadmat(activations[0])
    num_layers = 0
    layer_list = []
    for key in feat:
        print(key)
        if "__" in key:
            continue
        else:
            num_layers+=1
            layer_list.append(key)
    return num_layers,layer_list,num_condns

def get_features(feat_dir,layer_id,i):
    """
    to get activations of a particular DNN layer for a particular image

    Input:
    feat_dir: Directory containing activations generated using generate_features.py
    layer_id: layer name
    i: image index

    Output:
    flattened activations

    PS: This function is specific for activations generated using generate_features.py
    Write your own function in case you use different script to generate features.
    """
    activations = glob.glob(feat_dir + "/*" + ".mat")
    activations.sort()
    feat=sio.loadmat(activations[i])[layer_id]
    return feat.ravel()

def create_rdm(save_dir,feat_dir,dist):
    """
    Main function to create RDM from activations
    Input:
    feat_dir: Directory containing activations generated using generate_features.py
    save_dir : directory to save the computed RDM
    dist : dist used for computing RDM (e.g. 1-Pearson's R)

    Output (in submission format):
    The model RDMs for each layer are saved in
        save_dir/layer_name/submit_fMRI.mat to compare with fMRI RDMs
        save_dir/layer_name/submit_MEG.mat to compare with MEG RDMs
    """

    #get number of layers and number of conditions(images) for RDM
    num_layers,layer_list, num_condns = get_layers_ncondns(feat_dir)
    cwd = os.getcwd()

    # loops over layers and create RDM for each layer
    for layer in range(num_layers):
        os.chdir(cwd)
        #RDM is num_condnsxnum_condns matrix, initialized with zeros
        RDM = np.zeros((num_condns,num_condns))

        #save path for RDMs in  challenge submission format
        layer_id=layer_list[layer]
        RDM_dir = os.path.join(save_dir,layer_id)
        if not os.path.exists(RDM_dir):
            os.makedirs(RDM_dir)
        RDM_filename_meg = os.path.join(RDM_dir,'submit_meg.mat')
        RDM_filename_fmri = os.path.join(RDM_dir,'submit_fmri.mat')
        RDM_filename_meg_zip = os.path.join(RDM_dir,'submit_meg.zip')
        RDM_filename_fmri_zip = os.path.join(RDM_dir,'submit_fmri.zip')
        #RDM loop
        for i in tqdm(range(num_condns)):
            for j in tqdm(range(num_condns)):
                #get feature for image index i and j
                feature_i = get_features(feat_dir,layer_id,i)
                feature_j = get_features(feat_dir,layer_id,j)

                #compute distance 1-Pearson's R
                if dist == 'pearson':
                    RDM[i,j] = 1-np.corrcoef(feature_i,feature_j)[0][1]
                else:
                    print("The", dist, "distance measure not implemented, please request through issues")

        #saving RDMs in challenge submission format
        rdm_fmri={}
        rdm_meg={}
        rdm_fmri['EVC_RDMs'] = RDM
        rdm_fmri['IT_RDMs'] = RDM
        rdm_meg['MEG_RDMs_late'] = RDM
        rdm_meg['MEG_RDMs_early'] = RDM
        sio.savemat(RDM_filename_fmri,rdm_fmri)
        sio.savemat(RDM_filename_meg,rdm_meg)

        #creating zipped file for submission
        zipfmri = zipfile.ZipFile(RDM_filename_fmri_zip, 'w')
        zipmeg = zipfile.ZipFile(RDM_filename_meg_zip, 'w')
        os.chdir(RDM_dir)
        zipfmri.write('submit_fmri.mat')
        zipmeg.write('submit_meg.mat')
        zipfmri.close()
        zipmeg.close()





def main():

    RDM_distance_choice = ['pearson']

    parser = argparse.ArgumentParser(description='Creates RDM from DNN activations')
    parser.add_argument('-fd','--feat_dir', help='feature directory path', default = "./feats/vgg", type=str)
    parser.add_argument('-sd','--save_dir', help='RDM save directory path', default = "./rdms/vgg", type=str)
    parser.add_argument('-d','--distance', help='distance for RDMs', default = "pearson", choices=RDM_distance_choice)
    args = vars(parser.parse_args())

    # creates save_dir
    save_dir = os.path.join(args['save_dir'],args['distance'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # saves arguments used for creating RDMs
    args_file = os.path.join(save_dir,'args.json')
    with open(args_file, 'w') as fp:
        json.dump(args, fp, sort_keys=True, indent=4)

    feat_dir = args['feat_dir']
    dist = args['distance']

    #RDM function
    create_rdm(save_dir,feat_dir,dist)


if __name__ == "__main__":
    main()
