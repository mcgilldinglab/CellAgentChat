import numpy as np
import statsmodels.stats.multitest as multi
import os 
import permutations
from permutations import *
import pandas as pd
from scipy.stats import norm, gamma
import math



def get_distribution(fin, dist=0, scaled=False, pseudotime=False):
    print("Getting background distribution for each ligand-receptor pair")
    fin3 = {}
    i = 0
    j = []
    for key in fin:
        alist = fin[key]
        if not pseudotime:
            alpha, loc, scale = gamma.fit(alist, floc=0)
            if scaled and dist > 0:
                fin3[key] = (alpha/2, 0, (scale/dist)/2)
            else:
                fin3[key] = (alpha, 0, scale)
        else:
            #Pseudotime
            alpha, loc, scale = gamma.fit(alist, floc=min(alist)-0.00000001)
            fin3[key] = (alpha, loc, scale)
        i += 1
    return fin3



def save_distribution(fin, path = 'distribution.csv'):
    distribution = pd.DataFrame(fin)
    distribution.to_csv(path, index=False)
    print('Distribution Saved')
    
    
    
def load_distribution(file='distribution.csv'):
    distribution = pd.read_csv(file)
    fin = distribution.to_dict('list')
    print("Distribution loaded")
    return fin

    

def get_significant_lr_pairs(lr1, fin, cutoff=0.05):
    sig_lr = {}
    sig = 0
    unsig = 0
    pvalue = {}
    num_sig = {}
    pvalues2 = []
    out = []
    for cluster in lr1.keys():
        s = {}
        d1 = lr1[cluster]
        for key in d1:
            if key not in fin.keys():
                out.append(key)
                continue
            alpha, loc, scale = fin[key]
            val = d1[key]
            pval = 1-gamma.cdf(val, a=alpha, loc=loc, scale=scale)
            if math.isnan(pval):
                pval = 1
            pvalues2.append(pval)

    #print(max(pvalues2))
    #print(min(pvalues2))
    #print(sum(pvalues2)/len(pvalues2))
    corr = multi.multipletests(pvalues2, cutoff, method ="bonferroni")
    #print(len(corr[0]))

    i = 0
    for cluster in lr1.keys():
        d1 = lr1[cluster]
        #for all significant interactions
        d2 = {}
        #for pvalue
        d3 = {}
        #for num of significant Ligand-receptor pairs
        num = 0
        for key in d1:
            if key in out:
                continue
            if corr[0][i] == True:
                d2[key] = d1[key]
                d3[key] = corr[1][i]
                sig += 1
                num += 1
            else:
                unsig += 1
            i += 1
        if len(d2) != 0:
            sig_lr[cluster] = d2
            pvalue[cluster] = d3
            num_sig[cluster] = num

    print(str(sig)+" significant interactions")
    print("Percentage of significant interactions: " + str(sig/len(corr[0])*100))
    return sig_lr, pvalue, num_sig



def pvalues_threshold(pvalue):
    pvalue2 = {}
    for key in pvalue.keys():
        dict1 = pvalue[key]
        dict2 = {}
        for key2 in dict1.keys():
            val = dict1[key2]
            if val >0.025:
                dict2[key2] = "Small"
            elif val >0.01:
                dict2[key2] =  "Smaller"
            else:
                dict2[key2] = "Smallest"
        pvalue2[key] = dict2    
    return pvalue2



def get_interaction_matrix(clusters, num_sig):
    data = np.array(list(num_sig.values()))
    dict1 = {}
    for clus in clusters:
        dict2 = {}
        for clus2 in clusters:
            if str(clus)+"_"+ str(clus2) in num_sig.keys():
                dict2[clus2] = num_sig[str(clus)+"_"+str(clus2)]
            else:
                dict2[clus2] = 0
                
        dict1[clus] = dict2 
    return dict1


def save(sig_lr, pvalue, pvalue2, dict1, cluster_list, step, path='out',
    interaction_matrix='interaction_matrix.csv', sig_lr_pair='sig_lr_pairs.csv',
         pvalues_name='pvalues.csv', pvalues2_name='pvalues2.csv', cluster_names='cluster_names.csv'):
    #Interaction Matrix
    df1 = pd.DataFrame.from_dict(dict1, orient='index').transpose()
    df1.style.background_gradient(cmap="Reds")
    
    #Significant LR Pairs
    df2 = pd.DataFrame.from_dict(sig_lr, orient='index').fillna(0).transpose()
    
    #Pvalues
    df3 = pd.DataFrame.from_dict(pvalue, orient='index').fillna(0).transpose()
    
    #Pvalues2
    df4 = pd.DataFrame.from_dict(pvalue2, orient='index').fillna(0).transpose()
    
    #cluster names
    if cluster_list[0] != 'Names':
        cluster_list.insert(0, 'Names')
    clusters_list2 = cluster_list[:]
    clusters_list2.remove(False)
    df5 = pd.DataFrame(clusters_list2)
    
    
    #get top 25 significant lr pairs
    df2['total'] = df2.select_dtypes(np.number).gt(0).sum(axis=1)
    
    lrs = df2['total'].sort_values(ascending=False).head(25)
    l = list(lrs.index)
    
    df6 = df2.loc[l,:]
    df6.drop('total', inplace=True, axis=1)
    df6 = df6[df6.sum(0).sort_values(ascending=False)[:25].index]
    
    cols = df6.columns
    
    df7 = df3.loc[l,cols]
    df8 = df4.loc[l,cols]
    
    
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
       
    #Save Files
    df1.to_csv(path+"/"+str(step)+'_'+interaction_matrix)
    
    df6.to_csv(path+"/"+str(step)+'_new_'+sig_lr_pair)
    df7.to_csv(path+"/"+str(step)+'_new_'+pvalues_name)
    df8.to_csv(path+"/"+str(step)+'_new_'+pvalues2_name)
    
    df2.to_csv(path+"/"+str(step)+'_'+sig_lr_pair)
    df3.to_csv(path+"/"+str(step)+'_'+pvalues_name)
    df4.to_csv(path+"/"+str(step)+'_'+pvalues2_name)
    df5.to_csv(path+"/"+cluster_names)
    print("Saved files")

    
    
    
    
