import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

'''
Script to implement Hierarchical Risk Parity allocation. This has a closed-form solution. Most of the function have 
been borrowed from the work by TheRockXu (GitHub). Additional comments and a couple of minor
functions were written to build this sample script

@kenchan323
2021-1-30

References:
https://hudsonthames.org/portfolio-optimisation-with-portfoliolab-hierarchical-risk-parity/
https://github.com/TheRockXu/Hierarchical-Risk-Parity/blob/master/Hierarchical%20Clustering.ipynb

'''

def cov_to_corr_matrix(cov_mat):
    '''
    Convert a covariance matrix to a correlation matrix
    # Reference https://www.geeksforgeeks.org/convert-covariance-matrix-to-correlation-matrix-using-python/
    :param cov_mat: np array - a n by n 2d array representing a covariance matrix
    :return: np array - n by n correlation matrix representation of np array
    '''
    rows, cols = cov_mat.shape
    corr_mat = np.zeros((cols, cols))

    for i in range(cols):
        for j in range(cols):
            if i == j:
                corr_mat[i][j] = 1
            else:
                # note here that we are just normalizing the covariance matrix
                corr_mat[i][j] = cov_mat[i][j] / (np.sqrt(cov_mat[i][i]) * np.sqrt(cov_mat[j][j]))
    return corr_mat

def distance_calc(corr):
    '''
    Convert a Pearson correlation coefficient to a distance measure
    :param corr: float - Pearson correlation
    :return: float - distance
    '''
    if corr == 1:
        return 0
    else:
        return np.sqrt(0.5 * (1-corr))

def get_quasi_diag(link):
    '''
    Get the quasi diagonal of a distance matrix, which is represented as a linkage matrix
     # Reference https://github.com/TheRockXu/Hierarchical-Risk-Parity/blob/master/Hierarchical%20Clustering.ipynb
    :param link: 2d np array - linkage matrix as per scipy.cluster format
    :return: list - optimal ordering of leave to produce a quasi diagonal matrix
    TheRockXu (GitHub)
    '''
    # sort clustered items by distance
    link = link.astype(int)

    # get the first and the second item of the last tuple
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    # the total num of items is the third item of the last list
    num_items = link[-1, 3]

    # if the max of sort_ix is bigger than or equal to the max_items
    while sort_ix.max() >= num_items:
        # assign sort_ix index with 24 x 24
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # odd numers as index

        df0 = sort_ix[sort_ix >= num_items]  # find clusters

        # df0 contain even index and cluster index
        i = df0.index
        j = df0.values - num_items  #

        sort_ix[i] = link[j, 0]  # item 1

        df0 = pd.Series(link[j, 1], index=i + 1)

        sort_ix = sort_ix.append(df0)
        sort_ix = sort_ix.sort_index()

        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def get_cluster_var(cov, c_items):
    '''
    Get the cluster's variance
    :param cov: 2d array - the full covariance matrix of all stocks
    :param c_items: list - the idx's of the stocks in this particular sub-cluster
    :return: float - variance of the sub-cluster
    TheRockXu (GitHub) and refactored by kenchan323
    '''
    # matrix slice, only the subset of rows/col elements related to stocks in this cluster
    # calculate the inverse-variance portfolio
    sub_cov_matrix = cov.iloc[c_items, c_items]
    # For a quasi diagonal matrix, we assume the optimal is invsere variance allocation (which is true for diagonal matrix)
    ivp = 1. / np.diag(sub_cov_matrix) # reciprocal of the diagonal elements of the cov matrix
    ivp = ivp/ivp.sum() # divide them by the trace of the subset cov matrix
    w_ = ivp.reshape(-1, 1)
    cluster_variance = (np.array(w_.T) @ np.asmatrix(sub_cov_matrix) @ np.array(w_))
    cluster_variance = cluster_variance[0, 0] # flatten to get the only element
    return cluster_variance


def get_rec_bipart(cov, sort_ix):
    '''
    Perform recursive bisection on the full covariance matrix
    :param cov: 2d array - Covariance matrix of all stocks
    :param sort_ix: list - the order of the quasi-diagonal covariance matrix
    :return: list - the optimal weights of the HRP
    TheRockXu (GitHub)
    '''
    # compute HRP allocation
    # intialize weights of 1
    w = pd.Series(1, index=sort_ix)

    # intialize all items in one cluster
    c_items = [sort_ix]
    while len(c_items) > 0:
        # bisection
        """
        [[3, 6, 0, 9, 2, 4, 13], [5, 12, 8, 10, 7, 1, 11]]
        [[3, 6, 0], [9, 2, 4, 13], [5, 12, 8], [10, 7, 1, 11]]
        [[3], [6, 0], [9, 2], [4, 13], [5], [12, 8], [10, 7], [1, 11]]
        [[6], [0], [9], [2], [4], [13], [12], [8], [10], [7], [1], [11]]
        """
        c_items = [i[int(j):int(k)] for i in c_items for j, k in
                   ((0, len(i) / 2), (len(i) / 2, len(i))) if len(i) > 1]

        # now it has 2
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]  # cluster 1
            c_items1 = c_items[i + 1]  # cluter 2

            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            # Once we have the variances of the two sub-clusters, we use them to calculate the weights to be assigned
            # to each sub-cluster
            alpha = 1 - c_var0 / (c_var0 + c_var1)

            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

if __name__ == 'main':
    # dummy 4 by 4 covariance matrix
    cov = [[1.23, 0.375, 0.7, 0.3],
           [0.375, 1.22, 0.72, 0.135],
           [0.7, 0.72, 3.21, -0.32],
           [0.3, 0.135, -0.32, 0.52]]

    cov = np.array(cov)
    corr_mat = cov_to_corr_matrix(cov)

    # distance matrix
    df_dist = pd.DataFrame(corr_mat).applymap(lambda x: distance_calc(x))
    '''
    The different type of linkage options available
    Single Linkage – the distance between two clusters it the minimum distance between any two points in the clusters
    Complete Linkage – the distance between two clusters is the maximum of the distance between any two points in the clusters
    Average Linkage – the distance between two clusters is the average of the distance between any two points in the clusters
    Ward Linkage – the distance between two clusters is the increase of the squared error from when two clusters are merged
    '''
    link = linkage(df_dist, 'single', optimal_ordering=True)
    # linkage matrix.
    # See format:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering
    Z = pd.DataFrame(link)

    fig = plt.figure(figsize=(10, 5))
    dn = dendrogram(Z)

    sorted_index = get_quasi_diag(link)

    weights = get_rec_bipart(pd.DataFrame(data=cov), sorted_index)

    '''
    Sample output
    2    0.081317
    1    0.213957
    0    0.209404
    3    0.495322
    '''