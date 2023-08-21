import csv
import numpy as np
import scanpy as sc
from scipy.stats import norm, gamma
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import anndata
import matplotlib.pyplot as plt
import copy
import os


def load_db(adata, file = 'human_lr_pair.tsv'):
    print("Loading Database...")
    #human ligand universe
    lig_uni = {}
    #human receptor universe
    rec_uni = {}
    
    with open(file, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter='\t')
        i = 0
        for row in datareader:
            ligand = row[1]
            receptor = row[2]
            lig_keys = lig_uni.keys()
            rec_keys = rec_uni.keys()
            #skip headers
            if i == 0:
                i = i + 1
                continue
            #ligand
            if ligand in lig_keys and receptor in adata.var_names:
                list1 = lig_uni[ligand]
                list1.append(receptor)
            elif ligand in adata.var_names and receptor in adata.var_names:
                list1 = [receptor]
                lig_uni[ligand] = list1
            #receptor 
            if receptor in rec_keys and ligand in adata.var_names:
                list2 = rec_uni[receptor]
                list2.append(ligand)
            elif receptor in adata.var_names and ligand in adata.var_names:
                list2 = [ligand]
                rec_uni[receptor] = list2
    print("Database Loaded!")
    return lig_uni, rec_uni



def create_matrix(adata, rec_uni):
    #iterate through receptors
    tot = {}
    for rec in rec_uni.keys():
        lig = adata[:, rec_uni[rec]].X.toarray().sum() * 1000
        tot[rec] = lig
    
    val = adata[:,list(rec_uni.keys())].X.toarray()
    mean = np.average(val)
    std_dev = np.std(val)
    mat = []
    for i, row in enumerate(val):
        cdf = []
        for j in range(len(row)):
            if row[j] == 0:
                cdf.append(0)
            else:
                n = norm.cdf(row[j], loc=mean, scale=std_dev)
                curr_cell = adata[adata.obs_names[i], rec_uni[list(rec_uni.keys())[j]]].X.toarray().sum() *1000
                lig2 = tot[list(tot.keys())[j]] - curr_cell
                cdf.append(n*lig2)
        mat.append(cdf)

    mat = np.array(mat)
    return mat



def standardize(mat):
    smat = mat.transpose()
    for i, row in enumerate(smat):
        mean = np.mean(row)
        std = np.std(row)
        for j, val in enumerate(row):
            if std == 0:
                val2 = 0
            else:
                val2 = (val - mean)/std
            row[j] = val2
        smat[i] = row
    mat=smat.transpose()
    return mat


def normalize(mat):
    smat = mat.transpose()
    for i, row in enumerate(smat):
        mmin = min(row)
        mmax = max(row)
        for j, val in enumerate(row):
            if mmin == mmax:
                val2 = 0
            else:
                val2 = (val - mmin)/(mmax-mmin)
            row[j] = val2
        smat[i] = row
    mat=smat.transpose()
    return mat

    
 
class net(nn.Module):
    def __init__(self,input_size,output_size):
        super(net,self).__init__()
        self.l1 = nn.Linear(input_size, 1000)
        self.batchnorm = nn.BatchNorm1d(1000)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(1000, 1000)
        self.batchnorm = nn.BatchNorm1d(1000)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(1000,output_size)
        
    def forward(self,x):
        output = self.l1(x) 
        output = self.batchnorm(output)
        output = self.relu(output)
        output = self.l2(output)
        output = self.batchnorm(output)
        output = self.relu(output)
        output = self.l3(output)
        return output
    
    
class ExpDataset():
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length



    
def train(adata, rec_uni, path):
    print("Setting up model")
    mat = create_matrix(adata, rec_uni)
    mat = standardize(mat)
    c = adata.X.toarray()
    C = np.array(c)
    C = standardize(C)
    dataset = ExpDataset(mat,C)
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=128)
    model = net(mat.shape[1],C.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    epochs = 50
    
    #training
    print("Training model...")
    preds = []
    costval = []
    for j in range(epochs):
        #print("epoch: "+str(j))
        for i,(x_train,y_train) in enumerate(dataloader):
            optimizer.zero_grad()
            #prediction
            y_pred = model(x_train)
            #calculating loss
            cost = criterion(y_pred,y_train)
            #backprop
            cost.backward()
            optimizer.step()
        costval.append(cost)
    torch.save(model, path)
    print("Training complete!")
    return mat, c
    
    
def load_model(path):
    model = torch.load(path)
    return model


def perform_iteration(model, mat, C, criterion):
    dataset2 = ExpDataset(mat,C)
    dataloader2 = DataLoader(dataset=dataset2,shuffle=False, batch_size=128)
    for i,(x_train,y_train) in enumerate(dataloader2):
        y_pred = model(x_train)
    cost = criterion(y_pred,y_train)
    return y_pred, cost
        


def permutate_receptors(model, num_recs, ocost, mat, C, criterion, start=0, loss=[]):
    zeros = np.zeros(len(mat))
    for i in range(start, num_recs):
        #print("receptor: " + str(i))
        mat2 = copy.deepcopy(mat)
        mat2 = mat2.transpose()
        mat2[i] = zeros
        mat2 = mat2.transpose()
        mat2 = standardize(mat2)
        ypred, cost = perform_iteration(model, mat2, C, criterion)
        dcost = abs(cost - ocost)
        loss.append(dcost)
    return loss


def save_conversion_rate(conversion_rates, file='conversion_rates.txt'):
    with open(file, 'w') as fp:
        for values in conversion_rates:
            # write each item on a new line
            fp.write("%s\n" % values)
    print('Done')
    
    
def load_conversion_rate(file='conversion_rates.txt'):
    conversion_rates = []
    with open(file, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            conversion_rates.append(float(x))
    print("Conversion rates loaded")
    return conversion_rates
        

def feature_selection(model, mat, C, rec_uni):
    print("Performing feature selection to obtain conversion rates...")
    #one iteration of model
    criterion = nn.MSELoss()
    dataset2 = ExpDataset(mat,C)
    dataloader2 = DataLoader(dataset=dataset2,shuffle=False,batch_size=407)
    for i,(x_train,y_train) in enumerate(dataloader2):
        y_pred = model(x_train)
    cost = criterion(y_pred,y_train)
    
    losses = permutate_receptors(model, start=0, num_recs=len(rec_uni), mat=mat, C=C, ocost=cost, loss=[], criterion=criterion)
    
    lmin = min(losses)
    lmax = max(losses)
    l2 = [((val - lmin) / (lmax-lmin)).item() for val in losses]
    
    conversion_rates = [val if val > 0.4 else 0.4 for val in l2]
    print("Complete")
    return conversion_rates



def add_rates(conversion_rates, rec_uni):
    rates = dict(zip(list(rec_uni.keys()), conversion_rates))
    return rates



def get_target_genes(receptors, N, model, mat, C, rec_uni, adata, threshold=50):
    dict2 = {}
    y_pred, _ = perform_iteration(model, mat, C, nn.MSELoss())
    for rec in receptors:
        print("Getting Target Genes for receptor: " +rec)
        zeros = np.zeros(N)
        mat2 = copy.deepcopy(mat)
        mat2 = mat2.transpose()
        for i, rec2 in enumerate(rec_uni.keys()):
            if rec2 == rec:
                mat2[i] = zeros
        mat2 = mat2.transpose()
        mat2 = standardize(mat2)
        y_pred2, _ = perform_iteration(model, mat2, C, nn.MSELoss())
        y = abs(y_pred-y_pred2)
        y = y.cpu().detach().numpy()
        zz = y.mean(axis=0)
        
        ind = np.argpartition(zz, -threshold)[-threshold:]

        top50 = list(zz[ind])
        new=[]
        for i, gene in enumerate(adata.var_names):
            if i in ind:
                new.append(gene)
        #new
        dict2[rec] = (top50, new)
        
    return dict2
    


def plot_results(dict1, adata, path='figures'):
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
    for rec in dict1.keys():
        print("Plotting results for receptor: "+rec)
        # Barplot
        genes = dict1[rec][1]
        values = dict1[rec][0]
        values = [x*100 for x in values]
        genes2 = [y for _,y in sorted(zip(values,genes), reverse=True)]
        values.sort(reverse=True)
        plt.figure(figsize=(10,7))
        plt.xticks(rotation=90)
        plt.xlabel("Target Genes")
        plt.ylabel("Percent Change in Gene Expression")
        plt.ylim((min(values),max(values)))
        plt.bar(genes2, values)
        plt.savefig(path+'/'+rec+'_targets.pdf')
        
        
        # Matrix plot
        sc.pl.matrixplot(adata, genes2, groupby='cell_type', cmap='viridis',standard_scale='var',save=rec+'_matrix.pdf')
    print("plots saved to directory /"+path)



def block_receptors(adata, receptors, rec_uni, threshold=50):
    print("Blocked Receptor Analysis")
    model = load_model("model.pt")
    mat = create_matrix(adata, rec_uni)
    mat = standardize(mat)
    c = adata.X.toarray()
    C = np.array(c)
    C = standardize(C)
    print("Get original")
    dict1 = get_target_genes(receptors, len(adata.obs), model, mat, C, rec_uni, adata, threshold)
    plot_results(dict1, adata)
    
    
    
    