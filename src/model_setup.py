import csv
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import norm, gamma
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sparselinear as sl
import anndata
import matplotlib.pyplot as plt
import copy
import os
import random
import math
import scipy.stats



def load_db(adata, file = 'human_lr_pair.tsv', sep='\t'):
    print("Loading Database...")
    #human ligand universe
    lig_uni = {}
    #human receptor universe
    rec_uni = {}
    
    with open(file, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=sep)
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
                
    #List of LR pairs          
    lr_pairs = []
    for ligand in lig_uni.keys():
        for receptor in lig_uni[ligand]:
            lr_pairs.append(ligand+"_"+receptor)
    print("Database Loaded!")
    return lig_uni, rec_uni, lr_pairs



def delta_preprocesssing(adata, rec_uni, lig_uni):
    #Preprocessing to obtain the expressions of every lig/rec and coordinates for each spot
    lig_dist = {}
    lig_coords = {}
    rec_dist = {}
    rec_coords = {}
    genes = list(adata.var_names)

    x = list(adata.obs['x_coord'])
    y = list(adata.obs['y_coord'])
    for lig in lig_uni:
        j = genes.index(lig)
        exp = adata.X.toarray()[:, j]
        
        d = {"exp":exp, "x":x,"y":y}
        df2 = pd.DataFrame(data=d, index=cells)
        df2 = df2[df2.exp > 0.0]
        lig_dist[lig] = list(df2['exp'])
        lig_coords[lig] = np.column_stack((list(df2['x']), np.array(list(df2['y']))))
        

    for rec in rec_uni:
        j = genes.index(rec)
        exp = adata.X.toarray()[:, j]
        
        d = {"exp":exp, "x":x,"y":y}
        df2 = pd.DataFrame(data=d,index=cells)
        df2 = df2[df2.exp > 0.0]
        rec_dist[rec] = list(df2['exp'])
        rec_coords[rec] = np.column_stack((list(df2['x']), np.array(list(df2['y']))))
    
    return lig_dist, rec_dist, lig_coords, rec_coords


def calc_distance(pair, pos_l, pos_r, d_lig, d_rec, reg=1, reg_m=1, iters=100):
    lig, rec = pair.split("_")
    
    #unbalanced OT - forward direction 
    cost = ot.dist(pos_l, pos_r,metric='euclidean')
    cost = cost +1e-8
    distance_fwd = ot.sinkhorn_unbalanced2(d_lig,d_rec, cost/cost.max(), reg=reg, reg_m=reg_m, numItermax=iters)[0]
    
    #unbalanced OT - reverse direction
    cost2 = ot.dist(pos_r, pos_l,metric='euclidean')
    cost2 = cost2 +1e-8
    distance_rev = ot.sinkhorn_unbalanced2(d_rec,d_lig, cost2/cost2.max(), reg=reg, reg_m=reg_m, numItermax=iters)[0]
    
    #Final Distance
    gene_distance = (distance_fwd+distance_rev)/2
    
    return gene_distance


def load_tf_db(species, adata, rec_uni):
    if species == 'mouse':
        file1 = "../databases/TF_TG_mouse.csv"
        file2 = "../databases/KEGG_mouse.csv"
        file3 = "../databases/REACTOME_mouse.csv"
    if species == 'human':
        file1 = "../databases/TF_TG_human.csv"
        file2 = "../databases/KEGG_human.csv"
        file3 = "../databases/REACTOME_human.csv"
      
    #TF_TG dataframe
    df = pd.read_csv(file1)
    df = df.drop(columns="Unnamed: 0")
    
    #KEGG dataframe
    df2 = pd.read_csv(file2)
    df2 = df2.drop(columns="Unnamed: 0")
    
    #REACTOME dataframe
    df3 = pd.read_csv(file3)
    df3 = df3.drop(columns="Unnamed: 0")
    
    #df2 cleanup
    receptors = list(df2['receptor'])
    tf = list(df2['tf'])
    list1 = []
    list2 = []
    for i, rec in enumerate(receptors):
        l = rec.split(",")
        for rec2 in l:
            list1.append(rec2)
            list2.append(tf[i])
            
    df2 = pd.DataFrame({"Receptors":list1, "TF":list2})
    
    #df3 cleanup
    receptors = list(df3['receptor'])
    tf = list(df3['tf'])
    list1 = []
    list2 = []
    for i, rec in enumerate(receptors):
        l = rec.split(",")
        for rec2 in l:
            list1.append(rec2)
            list2.append(tf[i])
            
    df3 = pd.DataFrame({"Receptors":list1, "TF":list2})
    
    #Rec to TF dataframe
    rec_tf = pd.concat([df2, df3]).drop_duplicates()
    
    #TF universe - TF to downstream genes
    tf_uni = {}
    for _, row in df.iterrows():
        tf = row['TF']
        if tf in adata.var_names:
            gene = row['Target Gene']
            if tf in tf_uni.keys():
                gene_list = tf_uni[tf]
            else:
                gene_list = []
            if gene in adata.var_names and gene not in gene_list:
                gene_list.append(gene)
                tf_uni[tf] = gene_list
    
    #Rec to TF universe    
    rec_tf_uni = {}
    for _, row in rec_tf.iterrows():
        rec = row["Receptors"]
        tf = row["TF"]
        if rec in rec_tf_uni.keys():
            tf_list = rec_tf_uni[rec]
        else:
            tf_list = []
        if rec in rec_uni.keys() and tf in tf_uni.keys() and tf not in tf_list:
            tf_list.append(tf)
            rec_tf_uni[rec] = tf_list

    return tf_uni, rec_tf_uni



def create_masked_connections(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs):
    #inputs
    num_rec = len(rec_uni)
    num_lig = len(lig_uni)
    num_inputs = num_rec + num_lig
    l = list(rec_uni.keys())+list(lig_uni.keys())
    #inputs = adata[:, l].X.toarray()
    if scipy.sparse.issparse(adata.X):
        input_recs = adata[:, list(rec_uni.keys())].X.toarray()
        input_ligs = adata[:, list(lig_uni.keys())].X.toarray()
    else:
        input_recs = np.array(adata[:, list(rec_uni.keys())].X)
        input_ligs = np.array(adata[:, list(lig_uni.keys())].X)
    lig_averages = np.mean(input_ligs, axis=0)
    reshaped_averages = lig_averages.reshape(1, -1)
    repeated_averages = np.repeat(reshaped_averages, input_recs.shape[0], axis=0)
    inputs = np.concatenate((input_recs, repeated_averages), axis=1)


    #hidden layers
    num_lr_pairs = len(lr_pairs)
    num_tfs = len(tf_uni.keys())

    #output
    if scipy.sparse.issparse(adata.X):
        all_genes = adata.X.toarray()
    else:
        all_genes = np.array(adata.X)
    num_outputs = all_genes.shape[1]
    
    #indexes
    lr_index =dict(zip(lr_pairs, list(range(num_lr_pairs))))
    input_index = dict(zip(l,list(range(num_inputs))))
    tf_index = dict(zip(list(tf_uni.keys()),list(range(num_tfs))))
    gene_index = dict(zip(list(adata.var_names),list(range(num_outputs))))
    
    #Layer 1 connections
    mask = np.zeros((num_lr_pairs, num_inputs))
    for pair, i in list(lr_index.items()):
        lig, rec = pair.split("_")
        #ligand index
        j = input_index[lig]
        #receptor index 
        k = input_index[rec]
        #set connection between lig and receptor to node in LR pair layer
        mask[i][j] = 1
        mask[i][k] = 1
    mask = torch.nonzero(torch.tensor(mask)).T
    
    #Layer 2 Connections
    mask2 = np.zeros((num_tfs,num_lr_pairs))
    
    for pair, j in list(lr_index.items()):
        _, rec = pair.split("_")
        if rec in rec_tf_uni.keys():
            #get tfs for that receptor
            tf_list = rec_tf_uni[rec]
            for tf in tf_list:
                i = tf_index[tf]
                #set connection between LR pair with the receptor and TF
                mask2[i][j] = 1
        else:
            vector = np.ones(mask2.shape[0])
            mask2[:, j] = vector
            
    mask2 = mask2.T
    
    #Layer 3 Connections
    mask3 = np.zeros((num_outputs,num_tfs))
    for tf, j in list(tf_index.items()):
        #get downstream genes for each TF
        gene_list = tf_uni[tf]
        for gene in gene_list:
            i = gene_index[gene]
            #set connections between TF and downstream gene
            mask3[i][j] = 1

    fc_genes = []
    #for any downstream gene with no connection make them fully connected with previous layer
    for i in range(num_outputs):
        if not np.any(mask3[i]):
            vector = np.ones(len(mask3[i]))
            mask3[i] = vector
            fc_genes.append(adata.var_names[i])
    
    mask3 = mask3.T
    return inputs, all_genes, num_inputs, num_lr_pairs, num_tfs, num_outputs, mask, mask2, mask3


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


# Define custome autograd function for masked connection.

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.

        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

    
 
class net(nn.Module):
    def __init__(self,num_inputs, num_lr_pairs, num_tfs, output_size, mask1, mask2, mask3):
        super(net,self).__init__()
        self.layers = nn.Sequential(
            sl.SparseLinear(in_features=num_inputs, out_features=num_lr_pairs, connectivity=mask1),
            nn.BatchNorm1d(num_lr_pairs),
            nn.ReLU(),
            CustomizedLinear(torch.tensor(mask2), bias=None),
            nn.BatchNorm1d(num_tfs),
            CustomizedLinear(torch.tensor(mask3), bias=None),
            nn.ReLU()
        )
        
    def forward(self,x):
        output = self.layers(x)
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



    
def train(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs, path, epochs=50):
    print("Setting up model")
    o = create_masked_connections(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs)
    inputs, all_genes, num_inputs, num_lr_pairs, num_tfs, num_outputs, mask, mask2, mask3 = o
    dataset = ExpDataset(inputs, all_genes)
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=256)
    model = net(num_inputs, num_lr_pairs, num_tfs, num_outputs, mask, mask2, mask3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    
    #training
    print("Training model...")
    preds = []
    costval = []
    for j in range(epochs):
        print("epoch: "+str(j))
        for i,(x_train,y_train) in enumerate(dataloader):
            optimizer.zero_grad()
            #prediction
            y_pred = model(x_train)
            #calculating loss
            cost = criterion(y_pred,y_train)
            #backprop
            cost.backward()
            optimizer.step()
        print(cost)
        costval.append(cost)
    torch.save(model, path)
    print("Training complete!")
    return inputs, all_genes
    
    
def load_model(path):
    model = torch.load(path)
    return model


def perform_iteration(model, mat, C, criterion, batch = 256):
    dataset2 = ExpDataset(mat,C)
    dataloader2 = DataLoader(dataset=dataset2,shuffle=False, batch_size=batch)
    for i,(x_train,y_train) in enumerate(dataloader2):
        y_pred = model(x_train)
    cost = criterion(y_pred,y_train)
    return y_pred, cost
        


def permutate_receptors(model, start, num, ocost, mat, C, criterion, loss=[], perc=50):
    zeros = np.zeros(len(mat))
    for i in range(start, num):
        print(i)
        mat2 = copy.deepcopy(mat)
        mat2 = mat2.transpose()
        tmp = mat2[i]
        random.shuffle(tmp)
        new_arr_no_0 = tmp[tmp!=0.0]
        if perc == 100:
            tmp = zeros
        elif len(new_arr_no_0) != 0:
            scale = np.percentile(new_arr_no_0, 100-perc)
            if scale < 1:
                tmp = tmp*scale
            else:
                tmp = tmp/scale
        mat2[i] = tmp
        mat2 = mat2.transpose()
        ypred, cost = perform_iteration(model, mat2, C, criterion, batch = mat2.shape[0])
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
        

def feature_selection(model, mat, C, rec_uni, start=0, perc=50):
    print("Performing feature selection to obtain conversion rates...")
    #one iteration of model
    criterion = nn.MSELoss()
    dataset2 = ExpDataset(mat,C)
    dataloader2 = DataLoader(dataset=dataset2,shuffle=False,batch_size=407)
    for i,(x_train,y_train) in enumerate(dataloader2):
        y_pred = model(x_train)
    cost = criterion(y_pred,y_train)
    
    losses = permutate_receptors(model, perc, start=0, num=len(rec_uni), mat=mat, C=C, ocost=cost, loss=[], criterion=criterion)
    
    lmin = min(losses)
    lmax = max(losses)
    l2 = [val/mean for val in loss]
    conversion_rates=[]  
    for val in l2:
        if val > 1:
            conversion_rates.append(1)
        elif val < 0.4:
            conversion_rates.append(0.4)
        else: 
            conversion_rates.append(val)
    #l2 = [((val - lmin) / (lmax-lmin)).item() for val in losses]
    
    #conversion_rates = [val if val > 0.4 else 0.4 for val in l2]
    print("Complete")
    return conversion_rates



def add_rates(conversion_rates, rec_uni):
    rates = dict(zip(list(rec_uni.keys()), conversion_rates))
    return rates



def get_target_genes(receptors, N, model, mat, C, rec_uni, adata, perc=50, threshold=50):
    dict2 = {}
    y_pred, _ = perform_iteration(model, mat, C, nn.MSELoss(), batch = mat.shape[0])
    for rec in receptors:
        print("Getting Target Genes for receptor: " +rec)
        zeros = np.zeros(N)
        mat2 = copy.deepcopy(mat)
        mat2 = mat2.transpose()
        for i, rec2 in enumerate(rec_uni.keys()):
            if rec2 == rec:
        #new
                tmp = mat2[i]
        if perc != 100:
            random.shuffle(tmp)
            new_arr_no_0 = tmp[tmp!=0.0]
            scale = np.percentile(new_arr_no_0, 100-perc)
            if scale < 1:
                tmp = tmp*scale
            else:
                tmp = tmp/scale
            mat2[i] = tmp
        else:
            tmp = zeros
            mat2[i] = tmp
        mat2 = mat2.transpose()       
        """
                mat2[i] = zeros
        mat2 = mat2.transpose()
        mat2 = standardize(mat2)
        """
        y_pred2, _ = perform_iteration(model, mat2, C, nn.MSELoss(), batch = mat2.shape[0])
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
        plt.ylim((min(values),max(values)+0.05))
        plt.bar(genes2, values)
        plt.savefig(path+'/'+rec+'_targets.pdf')
        
        
        # Matrix plot
        sc.pl.matrixplot(adata, genes2, groupby='cell_type', cmap='viridis',standard_scale='var',save=rec+'_matrix.pdf')
    print("plots saved to directory /"+path)



def block_receptors(adata, receptors, rec_uni, lig_uni, net, perc=50, threshold=50):
    print("Blocked Receptor Analysis")
    model = load_model(net)
    #get NN inputs
    if scipy.sparse.issparse(adata.X):
        input_recs = adata[:, list(rec_uni.keys())].X.toarray()
        input_ligs = adata[:, list(lig_uni.keys())].X.toarray()
    else:
        input_recs = np.array(adata[:, list(rec_uni.keys())].X)
        input_ligs = np.array(adata[:, list(lig_uni.keys())].X)
    lig_averages = np.mean(input_ligs, axis=0)
    reshaped_averages = lig_averages.reshape(1, -1)
    repeated_averages = np.repeat(reshaped_averages, input_recs.shape[0], axis=0)
    inputs = np.concatenate((input_recs, repeated_averages), axis=1)

    #output
    if scipy.sparse.issparse(adata.X):
        all_genes = adata.X.toarray()
    else:
        all_genes = np.array(adata.X)
    
    print("Get original")
    dict1 = get_target_genes(receptors, len(adata.obs), model, inputs, all_genes, rec_uni, adata, perc, threshold)
    plot_results(dict1, adata)
    
    
    
    
