#!/usr/bin/env python
# This script computes the score for the comparison of Model RDM with
# the fMRI data 
#Input
#   -target_rdm.mat is the file that contains EVC and IT fMRI RDM matrices.
#   -submit_rdm.mat is the file that is the model RDM to be compared against the fMRI data submitted 
#Output
#   -EVC_corr and IT_corr is the correlation of model RDMs to EVC RDM and IT RDM respectively
#   -pval is the corresponding p-value showing the significance of the correlation

import os
import sys

import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy import io

#defines the noise ceiling squared correlation values for EVC and IT
nc78_EVC_R2 = 0.0640
nc78_IT_R2 = 0.0647
nc78_avg_R2 = (nc78_EVC_R2+nc78_IT_R2)/2.

#loads the input files if in .mat format
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def loadnpy(npyfile):
    return np.load(npyfile)


def load(data_file):
    root, ext = os.path.splitext(data_file)
    return {'.npy': loadnpy,
            '.mat': loadmat
            }.get(ext, loadnpy)(data_file)


def sq(x):
    return squareform(x, force='tovector', checks=False)

#defines the spearman correlation
def spearman(model_rdm, rdms):
    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]

#computes spearman correlation (R) and R^2, and ttest for p-value.
def fmri_rdm(model_rdm, fmri_rdms):
    corr = spearman(model_rdm, fmri_rdms)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]

def evaluate(submission, targets, target_names=['EVC_RDMs', 'IT_RDMs']):
    out = {name: fmri_rdm(submission[name], targets[name]) for name in target_names}
    out['score'] = np.mean([x[0] for x in out.values()])
    return out

#function that evaluates the RDM comparison.    
def test_fmri_submission():
    target_file = 'target_fmri.mat'
    submit_file = 'submit_fmri.mat'
    target = load(target_file)
    submit = load(submit_file)
    out = evaluate(submit, target)
    evc_percentNC = ((out['EVC_RDMs'][0])/nc78_EVC_R2)*100.      #evc percent of noise ceiling
    it_percentNC = ((out['IT_RDMs'][0])/nc78_IT_R2)*100.         #it percent of noise ceiling
    score_percentNC = ((out['score'])/nc78_avg_R2)*100.      #avg (score) percent of noise ceiling
    print('=' * 20)
    print('fMRI results:')
    print('Squared correlation of model to EVC (R**2): {}'.format(out['EVC_RDMs'][0]), ' Percentage of noise ceiling: {}'.format(evc_percentNC),'%', '  and significance: {}'.format(out['EVC_RDMs'][1]))
    print('Squared correlation of model to IT (R**2): {}'.format(out['IT_RDMs'][0]), '  Percentage of noise ceiling: {}'.format(it_percentNC),'%', '  and significance: {}'.format(out['IT_RDMs'][1]))
    print('SCORE (average of the two correlations): {}'.format(out['score']), '  Percentage of noise ceiling: {}'.format(score_percentNC),'%') 



# if __name__ == '__main__':
#     test_fmri_submission()

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
target_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("{} doesn't exist".format(submit_dir))

if os.path.isdir(submit_dir) and os.path.isdir(target_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit_file = os.path.join(submit_dir, 'submit_fmri.mat')
    target_file = os.path.join(target_dir, 'target_fmri.mat')
    scores_file = os.path.join(output_dir, 'scores.txt')

    # Read input files
    target = load(target_file)
    submit = load(submit_file)

    out = evaluate(submit, target)
    evc_percentNC = ((out['EVC_RDMs'][0])/nc78_EVC_R2)*100.      #evc percent of noise ceiling
    it_percentNC = ((out['IT_RDMs'][0])/nc78_IT_R2)*100.         #it percent of noise ceiling
    score_percentNC = ((out['score'])/nc78_avg_R2)*100.      #avg (score) percent of noise ceiling
    with open(scores_file, 'w') as f:
        f.write('evc: {}\n'.format(out['EVC_RDMs'][0]))
        f.write('it: {}\n'.format(out['IT_RDMs'][0]))
        f.write('score: {}\n'.format(out['score']))
        f.write('evc_percentNC: {}\n'.format(evc_percentNC))
        f.write('it_percentNC: {}\n'.format(it_percentNC))
        f.write('score_percentNC: {}\n'.format(score_percentNC)) 
