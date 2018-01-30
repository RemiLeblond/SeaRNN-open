from __future__ import division
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand
cdef extern from "stdlib.h":
    int RAND_MAX

cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_t, ndim=2] compute_counts(int n_samples,
                                                 int max_length,
                                                 np.ndarray[DTYPE_t, ndim=2] labels,
                                                 np.ndarray[DTYPE_t, ndim=2] predictions,
                                                 np.ndarray[DTYPE_t, ndim=1] lengths):

    """Computation of the cost matrix given the outputs of the rollin rollout strategy and the labels.
        Args:
            n_samples   : number of items
            max_length  : maximum sentence lenght
            labels      : labels (1D array that has been flatten from the
                          [n_samples x max_length] array)
            predictions : predicitons (1D array that has been flatten from the
                          [n_samples x max_length] array)
            lengths  : length of the items
                          [1D array of size n_samples]
        Returns:
            counts
    """

    cdef np.ndarray[DTYPE_t, ndim=2] counts = np.zeros([n_samples, 3], dtype=DTYPE)

    cdef int i, ptr_label, ptr_pred, len_sample, chunk_len_label, chunk_len_pred, chunk_pred, chunk_label

    for i in range(n_samples):
        len_sample = lengths[i]

        # labels variables
        ptr_label = 0
        chunk_len_label = 0
        chunk_label = 0

        # prediction variables
        ptr_pred = 0
        chunk_len_pred = 0
        chunk_pred = 0

        while ptr_label + chunk_len_label < len_sample or ptr_pred + chunk_len_pred < len_sample:
            if ptr_label + chunk_len_label <= ptr_pred + chunk_len_pred:
                # update the position of the pointer
                ptr_label = ptr_label + chunk_len_label
                # find the first beggining of chunk in label
                while labels[i, ptr_label] == 0 and ptr_label + 1 < len_sample:
                    ptr_label += 1
                # gets its ID
                chunk_label = labels[i, ptr_label]
                # resolve consistency
                if chunk_label % 2 == 0 and chunk_label > 0:
                    chunk_label -= 1

                # find its length
                chunk_len_label = 1
                while ptr_label + chunk_len_label < len_sample:
                    if labels[i, ptr_label + chunk_len_label] == chunk_label + 1:
                        chunk_len_label +=1
                    else:
                        break
                if chunk_label != 0:
                    counts[i, 0] += 1
            else:
                # update the position of the pointer
                ptr_pred = ptr_pred + chunk_len_pred

                # find the first beggining of chunk in pred
                while predictions[i, ptr_pred] == 0 and ptr_pred + 1 < len_sample:
                    ptr_pred +=1

                chunk_pred = predictions[i, ptr_pred]
                # resolve consistency
                if chunk_pred % 2 == 0 and chunk_pred > 0:
                    chunk_pred -= 1

                chunk_len_pred = 1

                while ptr_pred + chunk_len_pred < len_sample:
                    if predictions[i, ptr_pred + chunk_len_pred] == chunk_pred + 1:
                        chunk_len_pred += 1
                    else:
                        break

                if chunk_pred != 0:
                    counts[i, 1] +=1

            # check if the two chunks are matching
            if ptr_pred == ptr_label and chunk_len_pred == chunk_len_label and chunk_label == chunk_pred and chunk_label != 0:
                counts[i, 2] += 1

    return counts
