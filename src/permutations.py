import anndata
import numpy as np
import mesa
import abm
from abm import CellAgent
from abm import CellModel
import abm_pseudotime
from abm_pseudotime import CellAgent2
from abm_pseudotime import CellModel2
import os


def copy(adata):
    bdata = anndata.AnnData(
      X = adata.X,
      obs = adata.obs,
      var = adata.var)
    return bdata



def permutation_test(threshold, N, adata, lig_uni, rec_uni, rates,
                     dist=False, tau=2, rec_block=False):
    #Get average distance
    distance = 1
    if dist:
        model3 = CellModel(N, adata, lig_uni, rec_uni, rates, dist=True, delta=1, max_steps=1,
                    tau=tau, rec_block=rec_block, permutations=True)
        distance = model3.calc_normalized_dist()
    
    fin = {}
    avg = 0
    i = 0
    while avg < threshold:
        print("iteration: " + str(i))
        bdata = copy(adata)
        x = np.array(bdata.obs['cell_type'])
        np.random.shuffle(x)
        bdata.obs['cell_type'] = x
        model2 = CellModel(N, bdata, lig_uni, rec_uni, rates, dist=False, tau=tau,  max_steps=1,
                           rec_block=rec_block, permutations=True)
        for _ in range(1):
            model2.step()
            
        res = abm.get_lr_interactions2(model2)
        l= list(res.values())
        for dic in l:
            for key in dic.keys():
                if key in fin.keys():
                    list1 = fin[key]
                    list1.append(dic[key])
                    fin[key] = list1
                else:
                    list1 = []
                    list1.append(dic[key])
                    fin[key] = list1
                    
        #check average
        val = 0
        for key in fin.keys():
            val += len(fin[key])
        avg = val/len(fin.keys())
        print("Average Number of LR Pair Scores:" + str(avg))
        i += 1
    return fin, model2, distance




def permutation_test_pseudotime(threshold, N, adata, lig_uni, rec_uni, rates, bins = 10, dist=False,
        shift = 0, tau=2, noise=5, rec_block=False):
    fin = {}
    avg = 0
    i = 1
    while avg < threshold:
        print("iteration: " + str(i))
        bdata = copy(adata)
        x = np.array(bdata.obs['cell_type'])
        np.random.shuffle(x)
        bdata.obs['cell_type'] = x
        model2 = CellModel2(N=N, adata=bdata, lig_uni=lig_uni, rec_uni=rec_uni, rates=rates, bins=bins,
                            dist=dist, shift=shift, tau=tau, noise=noise, rec_block=rec_block, permutations=True)
        distance = model2.calc_dist()
        for _ in range(bins):
            model2.step() 
            
        res = abm_pseudotime.get_lr_interactions2(model2)
        l= list(res.values())
        for dic in l:
            for key in dic.keys():
                if key in fin.keys():
                    list1 = fin[key]
                    list1.append(dic[key])
                    fin[key] = list1
                else:
                    list1 = []
                    list1.append(dic[key])
                    fin[key] = list1
                      
        #check average
        val = 0
        for key in fin.keys():
            val += len(fin[key])
        avg = val/len(fin.keys())
        print("Average Number of LR Pair Scores: "+ str(avg))
        i += 1       
    return fin, model2, distance




