from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.space import MultiGrid 
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import numpy as np
import subprocess
import abm
from abm import CellAgent
from abm import CellModel
import bckground_distribution
from bckground_distribution import *




def CCI(N, adata, lig_uni, rec_uni, rates, distribution, clusters, dist, delta=1, max_steps=1,
        tau=2, noise=5, rec_block=False, plot_every_step=True, path='out', interaction_matrix='interaction_matrix.csv',
        sig_lr_pair='sig_lr_pairs.csv', pvalues_name='pvalues.csv', pvalues2_name='pvalues2.csv',
        cluster_names='cluster_names.csv', threshold=0.05):
     
    model = CellModel(N, adata, lig_uni, rec_uni, rates, max_steps, dist, delta,
                    tau, noise, rec_block)
    print("Calculating Interactions")
    for i in range(max_steps):
        print("Step: " +str(i))
        model.step()
        # Calculations 
        if plot_every_step or i == max_steps-1:
            print("Calculating Significant Interactions")
            lr1 = abm.get_lr_interactions2(model)
            sig_lr, pvalues, num_sig = get_significant_lr_pairs(lr1, distribution, threshold)
            pvalues2 = pvalues_threshold(pvalues)
            if clusters[0] == 'Names':
                clusters = clusters[1:]
            dict1 = get_interaction_matrix(clusters, num_sig)
            print("Saving Files")
            save(sig_lr, pvalues, pvalues2, dict1, clusters, i+1, path=path, interaction_matrix='interaction_matrix.csv',
        sig_lr_pair=sig_lr_pair, pvalues_name=pvalues_name, pvalues2_name=pvalues2_name,
        cluster_names=cluster_names)
            print("Plotting results")
            step = str(i+1)
            #Heatmap
            subprocess.call(f"Rscript heatmaps.R --interaction_file {path+'/'+ step +'_'+interaction_matrix} --cluster_names {path+'/'+cluster_names} --out {path+'/'}heatmap_step_{i+1}.pdf", shell=True)
            #Dotplot
            subprocess.call(f"Rscript dotplot.R --lr_file {path+'/'+ step +'_new_'+sig_lr_pair} --pvalues {path+'/'+ step +'_new_'+pvalues2_name} --out {path+'/'}dotplot_step_{i+1}.pdf", shell=True)
    
    return model

           
def receiving_score(model, path='out'):
    receiver = []
    cells = list(model.adata.obs_names)
    cell_type = list(model.adata.obs['cell_type'])
    for agent in model.schedule.agents:
        receiver.append(agent.num_r)
    df = pd.DataFrame({'Cell': cells, 'Cell Type': cell_type,
     'Receiving Score': receiver})
    df.to_csv(path+"/"+"cell_receiving_scores.csv")
    print("results saved to: "+path+"/"+"cell_receiving_scores.csv")
    return df


def plotting(path='out', step='1', interaction_matrix='interaction_matrix.csv', sig_lr_pair='sig_lr_pairs.csv',
             pvalues2_name='pvalues2.csv', cluster_names='cluster_names.csv'):
    print("Plotting results")
    #Heatmap
    subprocess.call(f"Rscript heatmaps.R --interaction_file {path+'/'+ step +'_'+interaction_matrix} --cluster_names {path+'/'+cluster_names} --out {path+'/'}heatmap_step_{step}.pdf", shell=True)
    #Dotplot
    subprocess.call(f"Rscript dotplot.R --lr_file {path+'/'+ step +'_new_'+sig_lr_pair} --pvalues {path+'/'+ step +'_new_'+pvalues2_name} --out {path+'/'}dotplot_step_{step}.pdf", shell=True)
    print("Plots saved")     
    
    
    