import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import seaborn as sns
import scipy


def create_anndata(gene_expression, meta, x_umap = None, coordinates = None, pseudotime = None):
    
    obs = gene_expression.index
    var = gene_expression.columns
    adata = anndata.AnnData(
        X = scipy.sparse.csr_matrix(gene_expression.values, dtype='float32'),
      obs = pd.DataFrame(index = obs),
      var = pd.DataFrame(index = var))
    #cell types
    cell_types = list(meta['cell_type'])
    adata.obs['cell_type'] = cell_types
    #Batch
    batch = list(meta['Batch']
    adata.obs['Batch'] = batch
    
    #spatial coordinates
    if isinstance(coordinates, pd.DataFrame):
        x_coords = list(coordinates['x'])
        y_coords = list(coordinates['y'])
        adata.obs['x'] = y_coords
        adata.obs['y'] = x_coords
        if max(x_coords) > 10000:
            adata.obs['x'] //= 1000
        if max(y_coords) > 10000:
            adata.obs['y'] //= 1000
    if isinstance(x_umap, pd.DataFrame):
        adata.obsm['X_umap'] = x_umap.to_numpy()
    
    #pseudotime
    if isinstance(pseudotime, pd.DataFrame):
        adata.obs['pseudotime'] = pseudotime['pseudotime']
          
    return adata


def expression_processor(adata: anndata, normalize = True):
    #Preprocess
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    if normalize: 
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.00005, max_mean=7, min_disp=0.05)
    adata = adata[:, adata.var.highly_variable]
    return adata


def plot_spatial(adata):
    sns_plot = sns.scatterplot(x='x',y='y',data=adata.obs, hue='cell_type')
    fig = sns_plot.get_figure()
    fig.savefig("spatial_plot.pdf")
    
    
