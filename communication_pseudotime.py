from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.space import MultiGrid 
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np
import subprocess
import abm_pseudotime
from abm_pseudotime import CellAgent2
from abm_pseudotime import CellModel2
import bckground_distribution
from bckground_distribution import *




def CCI_pseudotime(N, adata, lig_uni, rec_uni, rates, distribution, clusters, dist=False, bins = 10, delta=1, shift = 0,
        tau=2, noise=5, rec_block=False, path='out', interaction_matrix='interaction_matrix.csv',
        sig_lr_pair='sig_lr_pairs.csv', pvalues_name='pvalues.csv', pvalues2_name='pvalues2.csv',
        cluster_names='cluster_names.csv'):
    
    model = CellModel2(N, adata, lig_uni, rec_uni, rates, bins, dist, delta, shift,
                    tau, noise, rec_block)
    print("Calculating Interactions")
    for i in range(bins):
        
        model.step()
        
    # Calculations
    print("Calculating Significant Interactions")
    lr1 = abm_pseudotime.get_lr_interactions2(model)
    sig_lr, pvalues, num_sig = get_significant_lr_pairs(lr1, distribution)
    pvalues2 = pvalues_threshold(pvalues)
    if clusters[0] == 'Names':
        clusters = clusters[1:]
    dict1 = get_interaction_matrix(clusters, num_sig)
    print("Saving Files")
    step='final'
    save(sig_lr, pvalues, pvalues2, dict1, clusters, step, path=path, interaction_matrix='interaction_matrix.csv',
        sig_lr_pair=sig_lr_pair, pvalues_name=pvalues_name, pvalues2_name=pvalues2_name,
        cluster_names=cluster_names)
    print("Plotting results")
    #Heatmap
    subprocess.call(f"Rscript heatmaps.R --interaction_file {path+'/final_'+interaction_matrix} --cluster_names {path+'/'+cluster_names} --out {path+'/'}heatmap_step_{step}.pdf", shell=True)
    #Dotplot
    subprocess.call(f"Rscript dotplot.R --lr_file {path+'/' +'final_new_'+sig_lr_pair} --pvalues {path+'/'+'final_new_'+pvalues2_name} --out {path+'/'}dotplot_step_{step}.pdf", shell=True)
    print("Plots saved")      