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

cpdef tuple compute_bleu(int n_samples,
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
            bleu_score  : 1D int array of size n_samples containing bleu scores.
    """

    cdef np.ndarray[FLOATTYPE_t, ndim=1] bleu_scores = np.zeros(n_samples, dtype=FLOATTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] suffix_length = np.zeros(n_samples, dtype=DTYPE)

    cdef int i, j, k, s, gt_len, pr_len, n_max
    # 1-gram is hard coded here (TODO: make it more modular)
    n_max = 1
    cdef dict ref_counts, rev_ov_counts

    cdef LONG MAX_VOCAB_SIZE, hash_key

    # Vocab should not be bigger than 50K
    MAX_VOCAB_SIZE = 50000

    cdef np.ndarray[FLOATTYPE_t, ndim=1] correct = np.zeros(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] correct_overlap = np.zeros(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] counts = np.zeros(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] counts_overlap = np.zeros(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] tmp_bleu = np.ones(n_max, dtype=FLOATTYPE)
    cdef np.ndarray[FLOATTYPE_t, ndim=1] power = np.ones(n_max, dtype=FLOATTYPE)
    cdef FLOATTYPE_t bv_penalty, bleu

    # weights for the averaging (same as nltk for 4-grams)
    for j in range(n_max):
        power[j] = 0.25

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

            # test the prediction alone
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

        # start testing all admissible suffixes contained in GT
        # s is the lenght of the suffix
        for s in range(0, min(max_length-pr_len, gt_len)+1):
            for j in range(n_max):
                tmp_bleu[j] = 0
                if s >= j+1:
                    counts[j] += 1.0
                    # start with the common one (it is just adding a new letter)
                    # NB: this can be made faster by simply removing the last element
                    # from the hash and just adding a new first element
                    hash_key = 0
                    for k in range(j+1):
                        hash_key += (<LONG> pow(MAX_VOCAB_SIZE, k)) * (labels[max_length * i + gt_len-s+k] + 1)

                    # if this ngram is present update the counter
                    if hash_key in ref_counts.keys():
                        if ref_counts[hash_key] > 0:
                            correct[j] += 1.0
                            ref_counts[hash_key] -= 1.0

                if s > 0 and j >= 1 and pr_len+s >= j+1:
                    # go for the overlapping region
                    ref_ov_counts = dict(ref_counts)
                    counts_overlap[j] = 1
                    correct_overlap[j] = 0

                    hash_key = 0
                    for k in range(min(j, pr_len)):
                        hash_key += (<LONG> pow(MAX_VOCAB_SIZE, k)) * (predictions[max_length * i + max(0, pr_len-j)+k] + 1)
                    # add the first one in label
                    hash_key += (<LONG> pow(MAX_VOCAB_SIZE, min(j, pr_len))) * (labels[max_length * i + gt_len-s] + 1)

                    # check if more need to be added
                    for k in range(1, max(1, j-pr_len+1)):
                        hash_key += (<LONG> pow(MAX_VOCAB_SIZE, k+min(j, pr_len))) * (labels[max_length * i + gt_len-s+k] + 1)

                    # if this ngram is present update the counter
                    if hash_key in ref_ov_counts.keys():
                        if ref_ov_counts[hash_key] > 0:
                            correct_overlap[j] += 1.0
                            ref_ov_counts[hash_key] -= 1.0

                    for k in range(max(1, j-pr_len+1), min(s, j)):
                        # operate on the hash_key to remove first element
                        hash_key /= MAX_VOCAB_SIZE
                        # add new element
                        hash_key += (<LONG> pow(MAX_VOCAB_SIZE, j)) * (labels[max_length * i + gt_len-s+k] + 1)
                        counts_overlap[j] += 1.0
                        # if this ngram is present update the counter
                        if hash_key in ref_ov_counts.keys():
                            if ref_ov_counts[hash_key] > 0:
                                correct_overlap[j] += 1.0
                                ref_ov_counts[hash_key] -= 1.0

                if j==0 and correct[j] == 0 and correct_overlap[j] == 0:
                    # if there is no common unigram we can break directly and set score to 0
                    tmp_bleu[j] = 0
                else:
                    # compute bleu score for this ngram
                    tmp_bleu[j] = (correct[j] + correct_overlap[j]  + 1.) / (counts[j] + counts_overlap[j] + 1.)
                    tmp_bleu[j] = pow(tmp_bleu[j], power[j])

            # check if bv penalty applies or not
            bv_penalty = 1
            if pr_len + s < gt_len:
                if pr_len+s == 0:
                    bv_penalty = 0
                else:
                    bv_penalty = exp(1 - gt_len  / (pr_len+s))

            # aggregate bleu scores for each ngram into one
            bleu = bv_penalty
            for j in range(n_max):
                bleu *= tmp_bleu[j]

            if bleu > bleu_scores[i]:
                bleu_scores[i] = bleu
                suffix_length[i] = s

        # reinit counters and dict for next run
        for j in range(n_max):
            counts[j] = 0
            correct[j] = 0
            counts_overlap[j] = 0
            correct_overlap[j] = 0

        ref_counts.clear()

    return bleu_scores, suffix_length
