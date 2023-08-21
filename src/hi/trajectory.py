import scanpy as sc
import pandas as pd
import numpy as np
import slingshot
import matplotlib
from matplotlib import pyplot as plt


def entropy(n_array):
    n = n_array.toarray()
    N = np.sum(n)
    p = n/N
    return np.multiply(p, np.log(p)/np.log(N))


def sc_entropy(adata):
    entropy_list = []
    for spot in adata.obs_names:
        E = -1*sum(np.nan_to_num(entropy(adata[spot].X[0]))) 
        entropy_list.append(E[0])
    return np.argmax(entropy_list), entropy_list



def one_hot_encoding(adata, num_clusters):
    cluster_labels_onehot = np.zeros((adata.n_obs, num_clusters))
    clusters = set(list(adata.obs['cell_type']))
    tokenizer = {}
    index = 0
    for clust in clusters:
        tokenizer[clust] = index
        index += 1
    for i in range(adata.n_obs):
        c = adata.obs['cell_type'][i]
        cluster_labels_onehot[i][tokenizer[c]] = 1
    return cluster_labels_onehot, tokenizer


def save_pseudotime(adata, times, file):
    df = pd.DataFrame(list(zip(adata.obs_names, times)), columns=['cell','pseudotime'])
    df.to_csv(file)
    print("pseudotime values saved to " + file)



def get_trajectory(adata, file='pseudotime.csv'):
    num_clusters = len(set(list(adata.obs['cell_type'])))
    cluster_labels_onehot, tokenizer = one_hot_encoding(adata, num_clusters)
    root, _ = sc_entropy(adata)
    root_clust = tokenizer[adata.obs['cell_type'].to_list()[root]]
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    custom_xlim = (-12, 12)
    custom_ylim = (-12, 12)
    
    
    s = slingshot.Slingshot(adata.obsm['X_umap'], cluster_labels_onehot, start_node=root_clust, debug_level='verbose')
    s.fit(num_epochs=1, debug_axes=axes)
    times = np.array(s.plotter.sling.curves[0].pseudotimes_interp)
    
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    axes[0].set_title('Cell Type')
    axes[1].set_title('Pseudotime')
    s.plotter.curves(axes[0], s.curves)
    s.plotter.clusters(axes[0], labels=np.arange(s.num_clusters), s=4, alpha=0.5)
    s.plotter.clusters(axes[1], color_mode='pseudotime', s=5)
    print(times)
    save_pseudotime(adata, times, file)
    


def add_pseudotime(adata, file='pseudotime.csv'):
    df = pd.read_csv(file)
    adata.obs['pseudotime'] = list(df['pseudotime'])
    return adata
    
