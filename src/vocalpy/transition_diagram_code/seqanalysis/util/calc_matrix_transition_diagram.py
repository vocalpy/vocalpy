import glob
import re
import string

import numpy as np
import scipy.io as sio
from IPython import embed

from seqanalysis.util.logging import config_logging

log = config_logging()


def get_transition_matrix(bout, unique_labels):
    """
    Computes transition matrix and probability matrix for a given bout.

    Parameters:
    - bout (str): Bout sequence.
    - unique_labels (list): List of unique labels.

    Returns:
    - transM (numpy array): Transition matrix.
    - transM_prob (numpy array): Transition probability matrix.
    """

    transM = np.zeros((len(unique_labels), len(unique_labels)))
    transM_prob = np.zeros((len(unique_labels), len(unique_labels)))

    alphabet = {letter: index for index, letter in enumerate(unique_labels)}
    numbers = [alphabet[character] for character in bout if character in alphabet]

    for idx in range(len(numbers) - 1):
        transM[numbers[idx], numbers[idx + 1]] += 1

    # Normalize transition matrix
    transM_prob = (transM.T / np.sum(transM, axis=1)).T
    transM = transM.astype(int)

    return transM, transM_prob


def get_transition_matrix_befor_following_syl(bout, unique_lables):
    """
    Computes transition matrix and probability matrix for preceding and following syllables in a given bout.

    Parameters:
    - bout (str): Bout sequence.
    - unique_lables (list): List of unique labels.

    Returns:
    - transM_bsf (numpy array): Transition matrix.
    - transM_prob_bsf (numpy array): Transition probability matrix.
    """
    transM_bsf = np.zeros((len(unique_lables), len(unique_lables), len(unique_lables)))
    transM_prob_bsf = np.zeros(
        (len(unique_lables), len(unique_lables), len(unique_lables))
    )
    alphabet = {letter: index for index, letter in enumerate(unique_lables)}
    numbers = [alphabet[character] for character in bout if character in alphabet]

    for idx in range(1, len(numbers) - 1, 1):
        transM_bsf[numbers[idx - 1], numbers[idx], numbers[idx + 1]] += 1
        transM_prob_bsf[numbers[idx - 1], numbers[idx], numbers[idx + 1]] += 1

    transM_prob_bsf = (transM_prob_bsf.T / np.sum(transM_bsf, axis=1)).T
    transM_bsf = transM_bsf.astype(int)

    return transM_bsf, transM_prob_bsf


# def get_node_positions(source_target_list):
#     """
#     Computes node positions based on a source-target list.
#
#     Parameters:
#     - source_target_list (list): List of source-target pairs.
#
#     Returns:
#     - pos (numpy array): Array of node positions.
#     """
#     xpos = np.array([int(string[0]) for string in source_target_list])
#     ypos = np.zeros(len(xpos))
#     for i in range(len(np.unique(xpos))):
#         ypos[xpos == i] = [*range(len(xpos[xpos == i]))]
#
#     pos = np.column_stack((xpos, np.array(ypos)))
#
#     return pos


def get_node_matrix(matrix, edge_thres):
    """
    Calculates transition probabilities in percent and sets edges below a threshold to zero.

    Parameters:
    - matrix (numpy array): Transition probability matrix.
    - edge_thres (int): Threshold for edges to be set to zero.

    Returns:
    - matrix (numpy array): Updated matrix with probabilities in percent.
    """
    matrix = np.around(matrix, 2) * 100
    matrix = matrix.astype(int)
    matrix[matrix < edge_thres] = 0

    return matrix

