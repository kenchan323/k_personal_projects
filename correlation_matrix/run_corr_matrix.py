import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
try:
    import build_corr_matrix as build_corr
except ImportError:
    sys.path.insert(0, r"C:\dev\k_personal_projects\correlation_matrix")
    import build_corr_matrix as build_corr

def ultra_metric(rho):
    '''
    Turning correlation coefficient into a distance measure that satisfies strong triangle inequality
    :param rho:
    :return:
    '''
    return np.sqrt(2*(1-rho))

start_date = None
end_date = None
# start_date = dt.datetime(2019, 6, 1)
# end_date = dt.datetime(2019, 12, 1)

# Cluster count
n_clusters = 14
# Perplexity hyperparameter for TSNE
perp = 3

# Fetching a correlation matrix
df_corr = build_corr.parse_csv_returns(rho_matrix=True, start_date=start_date, end_date=end_date, use_name=True).dropna()
# Apply ultra metric calculation for distance matrix
df_corr_distance = df_corr.loc[:].apply(ultra_metric)
# 2D array
ndarray_distance = df_corr_distance.values
list_tickers = df_corr.index.values

dendro_linkage = "complete"
# "ward" cannot be used when a distance matrix is use (see below which comes from documentation page
'''
Methods ‘centroid’, ‘median’ and ‘ward’ are correctly defined only if Euclidean pairwise metric is used. 
If y is passed as precomputed pairwise distances, then it is a user responsibility to assure that these 
distances are in fact Euclidean, otherwise the produced result will be incorrect.
'''
links = linkage(squareform(ndarray_distance), method=dendro_linkage)
d = dendrogram(links, labels=[x[:8] for x in list_tickers])

if start_date == None and end_date == None:
    plt.title("DJ Dendrogram, Linkage={}, 20181226-20191225".format(dendro_linkage))
else:
    plt.title("DJ Dendrogram, Linkage={}, {}-{}".format(dendro_linkage,
                                                    start_date.strftime("%Y%m%d"),
                                                    end_date.strftime("%Y%m%d")))

# Linkage must be average if using a precomputed distance matrix
cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=dendro_linkage)
cluster_labels = cluster.fit_predict(ndarray_distance)

# Using TSNE to project a set of points based on a precomputed distance matrix
tsne_projection = TSNE(n_components=2, perplexity=perp, metric="precomputed", random_state=1).fit_transform(ndarray_distance)


ax_1 = plt.figure()
colour_palette = sns.color_palette("Paired", 36)
cluster_colours = [colour_palette[x] if x > 0
                   else (0.5, 0.5, 0.5) for x in cluster_labels]

for i, ticker in enumerate(list_tickers):
    x = tsne_projection[i][0]
    y = tsne_projection[i][1]
    plt.scatter(x, y, color=cluster_colours[i], s=500)
    plt.text(x + 0.3,  y + 0.3, ticker, fontsize=15)
    plt.axis("off")
    if start_date == None and end_date == None:
        plt.title("DJ Rho Clustering - "
                  "n_cluster={}, "
                  "Perplex={}, Metric={}, Linkage={}, 20181226-20191225 ".format(n_clusters, perp, "ultra", dendro_linkage))
    else:
        plt.title("DJ Rho Clustering - "
                  "n_cluster={}, "
                  "Perplex={}, Metric={}, Linkage={}, {}-{}".format(n_clusters, perp, "ultra", dendro_linkage,
                                                                         start_date.strftime("%Y%m%d"),
                                                                         end_date.strftime("%Y%m%d")))
plt.show()