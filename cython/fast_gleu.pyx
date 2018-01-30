from __future__ import division
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX

from libc.math cimport exp
from libc.math cimport pow

DTYPE = np.int
ctypedef np.int_t DTYPE_t


ctypedef unsigned long LONG

FLOATTYPE = np.float32
ctypedef np.float32_t FLOATTYPE_t

cpdef np.ndarray[FLOATTYPE_t, ndim=1] compute_gleu(int n_samples,
                                                   int max_length,
                                                   np.ndarray[DTYPE_t, ndim=1] labels,
                                                   np.ndarray[DTYPE_t, ndim=1] gt_lengths,
                                                   np.ndarray[DTYPE_t, ndim=1] predictions,
                                                   np.ndarray[DTYPE_t, ndim=1] pr_lengths):

    """Computation of the cost matrix given the outputs of the rollin rollout strategy and the labels.
        Args:
            n_samples   : number of items
            max_length  : maximum sentence lenght
            labels      : labels (1D array that has been flatten from the
                          [n_samples x max_length] array)
            gt_lengths  : length of the ground truth items
                          [1D array of size n_samples]
            predictions : predicitons (1D array that has been flatten from the
                          [n_samples x max_length] array)
            pr_lengths  : length of the prediction items
                          [1D array of size n_samples]
        Returns:
            gleu_score  : 1D int array of size n_samples containing gleu scores.
    """

    cdef np.ndarray[FLOATTYPE_t, ndim=1] gleu_scores = np.zeros(n_samples, dtype=FLOATTYPE)

    cdef int i, j, k, gt_len, pr_len, n_max
    # 4-gram is hard coded here (TODO: make it more modular)
    n_max = 4
    cdef dict ref_counts

    cdef LONG MAX_VOCAB_SIZE, hash_key

    # Vocab should not be bigger than 50K
    MAX_VOCAB_SIZE = 50000

    cdef np.ndarray[FLOATTYPE_t, ndim=1] correct = np.zeros(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] counts = np.zeros(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] gt_counts = np.zeros(n_max, dtype=FLOATTYPE)
    cdef FLOATTYPE_t tpfn, tpfp, tp

    # reference counts (dictionnary for GT ngrams)
    ref_counts = {}

    for i in range(n_samples):
        gt_len = gt_lengths[i]
        pr_len = pr_lengths[i]

        for j in range(n_max):
            if gt_len >= j+1:
                hash_key = 0
                for k in range(j+1):
                    hash_key += (<LONG> pow(MAX_VOCAB_SIZE, k)) * (labels[max_length * i + k] + 1)

                if hash_key in ref_counts.keys():
                    ref_counts[hash_key] += 1
                else:
                    ref_counts[hash_key] = 1

                gt_counts[j] += 1

                # compute the ref counts for j-grams
                for k in range(1 , gt_len-j):
                    # operate on the hash_key to remove first element
                    hash_key /= MAX_VOCAB_SIZE
                    # add new element
                    hash_key += (<LONG> pow(MAX_VOCAB_SIZE, j)) * (labels[max_length * i + k + j] + 1)

                    if hash_key in ref_counts.keys():
                        ref_counts[hash_key] += 1
                    else:
                        ref_counts[hash_key] = 1

                    gt_counts[j] += 1

            if pr_len >= j+1:
                counts[j] += 1
                # compute the ngrams counters
                hash_key = 0
                for k in range(j+1):
                    hash_key += (<LONG> pow(MAX_VOCAB_SIZE, k)) * (predictions[max_length * i + k] + 1)

                # if this ngram is present update the counter
                if hash_key in ref_counts.keys():
                    if ref_counts[hash_key] > 0:
                        correct[j] += 1.0
                        ref_counts[hash_key] -= 1.0

                for k in range(1, pr_len-j):
                    # operate on the hash_key to remove first element
                    hash_key /= MAX_VOCAB_SIZE
                    # add new element
                    hash_key += (<LONG> pow(MAX_VOCAB_SIZE, j)) * (predictions[max_length * i + k + j] + 1)
                    counts[j] += 1
                    # if this ngram is present update the counter
                    if hash_key in ref_counts.keys():
                        if ref_counts[hash_key] > 0:
                            correct[j] += 1.0
                            ref_counts[hash_key] -= 1.0

            if j==0 and correct[j] == 0:
                # if there is no common unigram we can break directly (all correct score will be 0)
                break
    

        # compute prec and recall
        tpfp = 0.0
        tpfn = 0.0
        tp = 0.0

        for j in range(n_max):
            tpfp += counts[j]
            tpfn += gt_counts[j]
            tp += correct[j]

        gleu_scores[i] = tp / max(tpfp, tpfn)

        # reinit counters and dict for next run
        for j in range(n_max):
            counts[j] = 0
            gt_counts[j] = 0
            correct[j] = 0

        ref_counts.clear()

    return gleu_scores
