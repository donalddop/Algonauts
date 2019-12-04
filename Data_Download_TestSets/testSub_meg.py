#!/usr/bin/env python
# This script computes the score for the comparison of Model RDM with
# the MEG data
#Input
#   -target_rdm.mat is the file that contains MEG RDM matrices from two time intervals.
#   -submit_rdm.mat is the file consisting of the model RDM to be compared against the MEG data 
#Output
#   -MEG_RDMs_early and MEG_RDMs_late is the correlation of model RDMs to early and late time intervals of the MEG RDM respectively
#   -pval is the corresponding p-value showing the significance of the correlation

import os
import sys

import h5py
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from scipy import io

#defines the noise ceiling squared correlation values for early and late time intervals
nc78_early_R2 = 0.3562
nc78_late_R2 = 0.4452 
nc78_avg_R2 = (nc78_early_R2+nc78_late_R2)/2.

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

#computes spearman correlation (R) and R^2, and ttest for p-value
def meg_rdm(model_rdm, meg_rdms):
    corr = np.mean([spearman(model_rdm, rdms) for rdms in meg_rdms], 1)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]

def evaluate_meg(submission, targets, target_names=['MEG_RDMs_early', 'MEG_RDMs_late']):
    out = {name: meg_rdm(submission[name], targets[name]) for name in target_names}
    out['score'] = np.mean([x[0] for x in out.values()])
    return out

#function that evaluates the RDM comparison. For Local machine usage. 
def test_meg_submission():
    target_file = 'target_meg.mat'
    submit_file = 'submit_meg.mat'
    target = load(target_file)
    submit = load(submit_file)
    out = evaluate_meg(submit, target)
    early_percentNC = ((out['MEG_RDMs_early'][0])/nc78_early_R2)*100.       #early percent of noise ceiling
    late_percentNC = ((out['MEG_RDMs_late'][0])/nc78_late_R2)*100.           #late percent of noise ceiling
    score_percentNC = ((out['score'])/nc78_avg_R2)*100.                       #avg (score) percent of noise ceiling
    print('=' * 20)
    print('MEG results:')
    print('Squared correlation of model to earlier time interval (R**2): {}'.format(out['MEG_RDMs_early'][0]), ' Percentage of noise ceiling: {}'.format(early_percentNC),'%', '  and significance: {}'.format(out['MEG_RDMs_early'][1]))
    print('Squared correlation of model to later time interval (R**2): {}'.format(out['MEG_RDMs_late'][0]), '  Percentage of noise ceiling: {}'.format(late_percentNC),'%', '  and significance: {}'.format(out['MEG_RDMs_late'][1]))
    print('SCORE (average of the two correlations): {}'.format(out['score']), '  Percentage of noise ceiling: {}'.format(score_percentNC),'%')


# if __name__ == '__main__':
#     test_meg_submission()

#Evaluating the RDM comparison on the server (CodaLab)
input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
target_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("{} doesn't exist".format(submit_dir))

if os.path.isdir(submit_dir) and os.path.isdir(target_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit_file = os.path.join(submit_dir, 'submit_meg.mat')
    target_file = os.path.join(target_dir, 'target_meg.mat')
    scores_file = os.path.join(output_dir, 'scores.txt')

    # Read input files
    target = load(target_file)
    submit = load(submit_file)

    out = evaluate_meg(submit, target)
    early_percentNC = ((out['MEG_RDMs_early'][0])/nc78_early_R2)*100.       #early percent of noise ceiling
    late_percentNC = ((out['MEG_RDMs_late'][0])/nc78_late_R2)*100.           #late percent of noise ceiling
    score_percentNC = ((out['score'])/nc78_avg_R2)*100.                       #avg (score) percent of noise ceiling
    with open(scores_file, 'w') as f:
        f.write('early: {}\n'.format(out['MEG_RDMs_early'][0]))
        f.write('late: {}\n'.format(out['MEG_RDMs_late'][0]))
        f.write('score: {}\n'.format(out['score']))
        f.write('early_percentNC: {}\n'.format(early_percentNC))
        f.write('late_percentNC: {}\n'.format(late_percentNC))
        f.write('score_percentNC: {}\n'.format(score_percentNC)) 