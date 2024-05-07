from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.space import MultiGrid 
from mesa.visualization.UserParam import *
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import pandas as pd
import scanpy as sc
import anndata
import math
import numpy as np
import random
import scipy.stats
from scipy.stats import norm, gamma
from itertools import combinations
import model_setup
from model_setup import *


def check_threshold(tensor1, tensor2, threshold):
    # Calculate the absolute difference between the two tensors
    diff1 = tensor1 - tensor2
    diff2 = tensor2 - tensor1
    
    # Check if each element difference is greater/less than the threshold
    greater_than_threshold = torch.greater(diff1, threshold)
    less_than_threshold = torch.greater(diff2, threshold)
    
    return greater_than_threshold, less_than_threshold




def gene_exp_preds(model_path, lig_uni, rec_uni, inputs, all_genes, threshold = 0.5, update=0.01):
    model = torch.load(model_path)
    
    #l = list(rec_uni.keys())+list(lig_uni.keys())
    #inputs2 = np.array(adata[:, l].X)
    #output
    #all_genes2 =np.array(adata.X)
    #print(all_genes2)
    
    dataset2 = ExpDataset(inputs, all_genes)
    dataloader2 = DataLoader(dataset=dataset2,shuffle=False,batch_size=len(all_genes))
    
    for i,(x_train,y_train) in enumerate(dataloader2):
        #print(y_train)
        y_pred = model(x_train)
        #print(y_pred)
        for i, cell1 in enumerate(y_pred):
            #print(i)
            t1 = y_train[i]

            greater_than_threshold, less_than_threshold = check_threshold(t1, cell1, threshold)

            #Increase expression
            indices = torch.where(greater_than_threshold)
            #print(len(indices[0]))
            #print("")
            for index in indices:
                #print(index)
                all_genes[i][index] += update

            #decrease expression
            indices2 = torch.where(less_than_threshold)
            for index in indices2:
                all_genes[i][index] -= update
                
    all_genes = np.clip(all_genes, 0, None)
            
    return all_genes



def get_protein_choices(rec_uni):
    #proteins
    proteins = []
    for rec in list(rec_uni.keys()):
        proteins.append(rec)

    proteins.sort()
    proteins.insert(0, False)
    return proteins



def get_lr_choices(rec_uni, lig_uni):
    #L-R pairs
    pairs = []
    for lig in list(lig_uni.keys()):
        rlist = lig_uni[lig]
        for rec in rlist:
            pair = lig +str("-")+rec
            if pair not in pairs:
                pairs.append(pair)
    for rec in list(rec_uni.keys()):
        llist = rec_uni[rec]
        for lig in llist:
            pair = lig +str("-")+rec
            if pair not in pairs:
                pairs.append(pair)
                
    pairs.sort()
    pairs.insert(0,False)
    return pairs



def get_cluster_choices(adata):
    clusters = list(set(adata.obs['cell_type']))
    clusters.insert(0, False)
    return clusters


def parameters(adata, lig_uni, rec_uni, rates, clusters, pairs, proteins):
    model_params = {
    'N': NumberInput(
        'Number of agents', adata.shape[0]),
    'adata': adata,
    'lig_uni': lig_uni,
    'rec_uni': rec_uni,
    'rates': rates,
    'max_steps': Slider(
        'Max number of steps', 1, 1, 100, 5),
    'delta': Slider(
        'Delta', 1, 0, 10, 0.1),
    'tau': Slider(
        'Tau', 2, 1, 10, 1),
    'rec_block': Choice(
        'Block Receptor', value=False, choices= proteins), 
    'protein_choice': Choice(
        'Receptor', value=proteins[0], choices= proteins), 
    'lr_choice': Choice(
        'L-R Pair', value=pairs[0], choices= pairs),
    'dist': Checkbox(
        'Distance', value=True),
    'sender': Choice(
        'Sender', value=clusters[0], choices= clusters),
    'receiver': Choice(
        'Receiver', value=clusters[0], choices= clusters),
    'noise': Slider(
        'Gaussian Noise Percentage', 5, 0, 50, 1),
    'text': StaticText("This is a descriptive textbox")
    }
    return model_params



def get_interactions(model):
    df1 = pd.DataFrame.from_dict(model.results, orient='index').transpose()
    df1 = df1.rename({0:'Sum'})
    return df1
    
    
def get_lr_interactions(model):
    df2 = pd.DataFrame.from_dict(model.results2,orient='index').fillna(0).transpose()
    return df2


def get_lr_interactions2(model):
    return model.results2


def get_avg_dist(model):
    return model.avg_dist


class CellAgent(Agent):

    def __init__(self, unique_id, model, clust, exp, batch):
        super().__init__(unique_id, model)
        #Whether the cell is stationary or can move
        self.mobile = True
        self.cluster = clust
        self.expression = exp
        self.slice = batch
        #Visualization Tracking Fields
        self.num_r = 0
        self.iqr = "very_low"
        
        
    def Message(self):
        rates2 = {}
        rec_list = [(key, self.expression[key]) for key in self.expression.keys() if key in self.model.rec_uni.keys() 
                            and self.expression[key] > 0.0]
        if len(rec_list) != 0:
            #Receptor cdf - Receiving Rate
            unzipped_rec = list(zip(*rec_list)) 
            rr = np.array(unzipped_rec[1])/np.max(np.array(unzipped_rec[1]))
            if rr.min() < 0.3:
                rr = (np.array(unzipped_rec[1])/np.max(np.array(unzipped_rec[1]))*0.7)+0.3
            for i, val in enumerate(unzipped_rec[0]):
                rates2[val] = self.model.rates[val] * rr[i]
            
            #mean = np.average(unzipped_rec[1])
            #std_dev = np.std(unzipped_rec[1])
            #for i, val in enumerate(unzipped_rec[1]):
                #Add Receiving Rate to Rates
                #if std_dev != 0:
                    #rates2[unzipped_rec[0][i]] = self.model.rates[unzipped_rec[0][i]] * norm.cdf(val, loc=mean, scale=std_dev)
                #else:
                    #rates2[unzipped_rec[0][i]] = self.model.rates[unzipped_rec[0][i]]
        clust1 = self.cluster
        #Ligand receptor interactions
        #ligand expression
        cell_ligands = self.model.ligs
        if self.model.dist_param != 0:
            distances = self.model.distances
            c = distances[self]
        #calc distances
        for clust2 in cell_ligands.keys():
            ckey = str(clust2)+ "_" + str(clust1)
            #Get outputs
            if ckey in self.model.output.keys():
                cnum = self.model.output[ckey]
                cdict = self.model.output2[ckey]
            else:
                cnum = 0
                cdict = {}
            #ligand expression for sending cluster
            lig_exp = cell_ligands[clust2]
            if self.model.dist_param != 0:
                dist2 = c[clust2]
            if len(rec_list) != 0:
                #Each receptor
                for rec in unzipped_rec[0]:
                    #Possible ligands for each receptor
                    poss_ligs = self.model.rec_uni[rec]
                    for lig in poss_ligs:
                        #Expression of each ligand
                        exp = lig_exp[lig]
                        #Calculate ligand score
                        if self.model.dist_param != 0:
                            if not (isinstance(self.model.delta, int) or isinstance(self.model.delta, float)):
                                lig_delta = self.model.delta[lig]
                            else:
                                lig_delta = self.model.delta
                            new_dist2 = [1/(d**(self.model.dist_param*lig_delta))if d != 0 else 0 for d in dist2]
                            b_length = np.count_nonzero(new_dist2)
                            lig_score = np.multiply(exp, new_dist2).sum() / b_length 
                        else:
                            lig_score = exp
                    
                        if lig_score != 0:
                            #L-R string
                            lr_str = lig + "-" + rec
                            #Multiply rates
                            final = lig_score * rates2[rec]
                            if (self.model.protein_choice == False or self.model.protein_choice == rec) and \
                            (self.model.sender == clust2 or self.model.sender == False) and \
                            (self.model.receiver == clust1 or self.model.receiver == False) and \
                            (self.model.lr_choice == False or self.model.lr_choice == lr_str):
                                self.num_r += final

                            cnum = cnum + final
                            #sum of specific LR pairs between clusters
                            if lr_str in cdict.keys():
                                lr_val = cdict[lr_str]
                                lr_val.append(final)
                                cdict[lr_str] = lr_val  
                            else:
                                cdict[lr_str] = [final]
            #Save interactions
            self.model.output[ckey] = cnum
            self.model.output2[ckey] = cdict
        #For Animation
        if self.model.receiver == clust1 or self.model.receiver == False:
            self.model.rec_score.append(self.num_r)
            
        
    def step(self):
        #Agent's step
        #print("Hi, I am agent " + str(self.unique_id) + ", cluster is " + str(self.cluster))
        self.Message()
  


class CellModel(Model):
    """
    CellAgentChat Parameters Help

    Number of Agents: The total count of cells utilized in the simulation.
    Max number of steps: The duration of the simulation expressed in iterations.
    Delta: Influences the degree of cell-to-cell distance (default=1). For long-range mode, a delta value less than 1 is used, while for short-range mode, a delta value greater than 1 is employed.
    Tau: Represents the degree of freedom for distance (default=2).
    Block Receptor: Specifies the receptor to be obstructed.
    Receptor Track: Designates the receptor to be tracked by the animation. Only interactions involving the selected receptor will be displayed on the screen.
    L-R Pair: Identifies the L-R pair to be tracked by the animation. Only interactions involving the chosen L-R pair will be shown on the screen.
    Distance: Distance mode setting.
    Sender: Refers to the cell type responsible for sending ligands. Only interactions where the ligands originate from the specified sending cell type will be displayed on the screen.
    Receiver: Refers to the cell type responsible for receiving ligands. Only interactions where ligands are received by the specified receiving cell type will be shown on the screen.
    Gaussian Noise Percentage: The percentage of Gaussian noise added to ligand expression to enhance the dynamism of the simulations (default=5%).
    """

    def __init__(self, N, adata, lig_uni, rec_uni, rates, max_steps, dist, delta=1,
                 tau=2, diff = 1, rec_block=False, protein_choice=False, lr_choice=False,sender=False,receiver=False,
                 permutations=False, net = None):
        #Mandatory Inputs
        self.num_agents = N
        self.adata = adata
        self.lig_uni = lig_uni
        self.rec_uni = rec_uni
        self.rates = rates
        self.max_steps = max_steps
        self.dist_on = dist
        #Other mandatory fields
        self.running = True
        self.genes = [gene for gene in adata.var_names if gene in rec_uni.keys() or 
                     gene in lig_uni.keys()]
        self.grid = None
        self.schedule = BaseScheduler(self)
        self.curr_step = 0
        
        #Optional Fields
        self.delta = delta
        self.diffusion=diff
        self.protein_choice = protein_choice
        self.lr_choice = lr_choice
        self.dist_param = tau    #tau
        self.rec_block = rec_block
        self.sender=sender
        self.receiver=receiver
        #self.noise = 0
        #For permutations
        self.permutations = permutations
        #Neural Network model
        self.net = net
        self.all_genes = None
        
        #avg distance
        self.avg_dist = 0
        
        #Agent messaging outputs
        #sum of all ligand interactions between clusters
        self.output = {}
        #sum of specific LR pairs between clusters
        self.output2 = {}
        #tmp outputs for mult steps - same format as output and output2
        self.output3 = {}
        self.output4 = {}
        
        #Results
        self.results = {}
        self.results2 = {}
    
        #Helper Fields
        self.ligs = 0         #{cluster:{lig_name:[exp1,...,expn]}} 
        self.distances = 0    #{cell1: {cluster1: [1/dist1, ..., 1/distn]}}
        self.clusters = {}    #Dictionary of cells in each cluster
        self.rec_score = []   #For animation
        self.tokenizer = {}
    
        
        #create grid
        if self.dist_on:
            self.grid = MultiGrid(75,55, True) 
        else:
            self.grid = MultiGrid(75,55, True)
    
        #tokenizer
        cts = list(set(adata.obs['cell_type']))
        for i, clust in enumerate(cts):
            self.tokenizer[str(clust)] = i
        #self.tokenizer = {'Tumor': 0, 'Stromal Normal': 1, 'T cell': 2, 'Adipocytes': 3, 'TRAC+ Cells': 4, 'Plasma Cells': 5, 'B Cells': 6, 'Plasmacytoid Dendritic': 7, 'Tumor Associated Stromal': 8, 'Endothelial': 9, 'Myoepithelial': 10, 'Epithelial': 11, 'Macrophage': 12, 'Transitional Cells': 13, 'Mast Cells': 14}
        #print(self.tokenizer)
            
        #all gene expression of all cells
        if scipy.sparse.issparse(adata.X):
            self.all_genes = adata.X.toarray()
        else:
            self.all_genes = np.array(adata.X)
            
        # Create agents
        for i in range(self.num_agents):
            
            #Cluster
            clust = str(adata.obs.loc[adata.obs_names[i], 'cell_type'])
            
            batch = str(adata.obs.loc[adata.obs_names[i], 'Batch'])

            #gene expression
            if scipy.sparse.issparse(adata.X):
                val = self.adata[adata.obs_names[i], self.genes].X.toarray()
                
            else:
                val = self.adata[adata.obs_names[i], self.genes].X
            val = list(val[0])
            exp = dict(zip(self.genes, val))
                    
            a = CellAgent(i, self, clust, exp, batch)
            self.schedule.add(a)
            
            #coordinates
            if self.dist_on:
                x, y = (adata.obs.loc[adata.obs_names[i], 'x'], adata.obs.loc[adata.obs_names[i], 'y'])
                self.grid.place_agent(a, (x, y))               
            else:
                x = random.randint(0,40)
                y = random.randint(0,40)
                self.grid.place_agent(a, (x, y))
                self.dist_param = 0
            
            #Add cells to clusters
            if clust in self.clusters.keys():
                cell_list = self.clusters[clust]
                cell_list.append(a)
                self.clusters[clust] = cell_list
            else:
                self.clusters[clust] = [a]
            
            
    #Function to modify database      
    def modify_db(self):
        if self.rec_block == False:
            for rec in self.rec_block:
                self.rates[rec] = 0
     
    
    def get_clusters(self):
        return self.clusters
    
    
    def get_dist(self):
        #{cell1: {cluster1: [1/dist1, ..., 1/distn]}}
        new_dict = {}
        clusters = self.get_clusters()
        for cell in self.schedule.agents:
            adict = {}
            for clust in clusters.keys():
                cells = clusters[clust]
                distances = []  #for distances for each cell and celltype
                for cell2 in cells:
                    if cell.slice == cell2.slice:
                        d = math.sqrt((cell.pos[0]-cell2.pos[0])**2 + (cell.pos[1]-cell2.pos[1])**2)#**(self.dist_param*self.delta)
                        if d == 0:
                            d = 1
                    else:
                        d = 0
                    #d = 1/d
                    distances.append(d)
                adict[clust] = distances
            new_dict[cell] = adict
        return new_dict
    
    
    
    
    def calc_ligands(self):
        #{cluster:{lig_name:[exp1,...,expn]}}
        new_dict = {}
        clusters = self.get_clusters()
        for clust in clusters.keys():
            cells = clusters[clust]
            ligd = {}
            for lig in self.lig_uni.keys():
                expl = []
                for cell in cells:
                    #Gaussian Noise
                    #std = cell.expression[lig]*1000 * (self.noise/100)
                    #exp = np.random.normal(cell.expression[lig]*1000,std)
                    exp = cell.expression[lig]
                    expl.append(exp)
                if self.dist_param == 0:
                    expl = sum(expl) / len(expl)
                ligd[lig] = expl
            new_dict[clust] = ligd
        return new_dict
    

    def dist(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    
    def calc_dist(self):
        points = [cell.pos for cell in self.schedule.agents]
        distances2 = [self.dist(p1, p2) for p1, p2 in combinations(points, 2)]
        avg_distance = sum(distances2) / len(distances2)
        return avg_distance
    
        

    def step(self):
        """Advance the model by one step."""
        self.rec_score = []
        if self.dist_param != 0 and self.curr_step == 0:
            self.avg_dist = self.calc_dist()
            self.distances = self.get_dist()
            print("Average Distance: "+str(self.avg_dist))
        self.ligs = self.calc_ligands()
        self.modify_db()

        self.schedule.step()
        
        #Save to results
        for cpair in self.output.keys():
            splitted = cpair.split('_')
            c2 = splitted[1]
            cdict = self.output2[cpair]
            #
            if cpair in self.output4.keys():
                cdict2 = self.output4[cpair]
            else:
                cdict2 = {}
            num = len(self.clusters[c2])
            for lr in cdict.keys():
                val = sum(cdict[lr]) / num
                #for multiple steps
                if self.curr_step != 0:
                    #print('hi')
                    prev = self.output4[cpair][lr]
                    prev2 = self.output3[cpair]
                else: 
                    prev = 0
                    prev2 = 0
                cdict[lr] = (val + prev)/ (self.curr_step+1)
                cdict2[lr] = (val + prev)
            self.output4[cpair] = cdict2
            self.output3[cpair] = (sum(list(self.output2[cpair].values())) + prev2)
            self.results2[cpair] = cdict
            self.results[cpair] = (sum(list(self.output2[cpair].values())) + prev2) / (self.curr_step+1)
            
        self.output = {}
        self.output2 = {}
        if not self.permutations:
            #Calculating ligand received IQR
            score2 = [x / (self.curr_step+1) for x in self.rec_score]
            score2 = sorted(score2)
            if len(score2) != 0:
                b_size = len(score2) // 6
                high = b_size * 5
                med_high = b_size * 4
                med_low = b_size * 3
                low = b_size * 2
                very_low = b_size * 1
                for agent in self.schedule.agents:
                    received = agent.num_r / (self.curr_step+1)
                    if agent.cluster == self.receiver or self.receiver == False:
                        if received <= score2[very_low]:
                            agent.iqr = 'very_low'
                        elif received < score2[low]:
                            agent.iqr = 'low'
                        elif received < score2[med_low]:
                            agent.iqr = 'med_low'
                        elif received < score2[med_high]:
                            agent.iqr = 'med_high'
                        elif received < score2[high]:
                            agent.iqr = 'high'
                        else:
                            agent.iqr = 'very_high'
                    else:
                        agent.iqr = 'very_low'
        self.curr_step += 1
        #dynamic gene expression update
        if self.max_steps != 1:
            #get lig rec input for NN
            recc = [i for i, gene in enumerate(self.adata.var_names) if gene in self.rec_uni.keys()]
            ligg = [i for i, gene in enumerate(self.adata.var_names) if gene in self.lig_uni.keys()]
            input_recs = self.all_genes[:, recc]
            input_ligs = self.all_genes[:, ligg]
            lig_averages = np.mean(input_ligs, axis=0)
            reshaped_averages = lig_averages.reshape(1, -1)
            repeated_averages = np.repeat(reshaped_averages, input_recs.shape[0], axis=0)
            inputs = np.concatenate((input_recs, repeated_averages), axis=1)
            self.all_genes = gene_exp_preds(self.net, self.lig_uni, self.rec_uni, inputs, self.all_genes, threshold = 0.5, update=0.01)
            
            #update lig, rec expression for each cell
            genes_ind = [i for i, gene in enumerate(self.adata.var_names) if gene in self.rec_uni.keys() or 
                     gene in self.lig_uni.keys()]
            for i, cell in enumerate(self.schedule.agents):
                val = self.all_genes[i, genes_ind]
                val = list(val)
                exp = dict(zip(self.genes, val))
                cell.expression = exp
            
            
        if self.max_steps == self.curr_step:
            if self.rec_block != False:
                block_receptors(self.adata, self.rec_block, self.rec_uni, self.lig_uni, self.net)
            self.running = False



#visualization   
def agent_portrayal(agent):
    #Colour cell based on cluster
    colours = ['violet', 'brown', 'pink','red','green', 'orange', 'grey', 'teal', 'navy', 'blue', 'magenta', 'cyan', 'yellow','black','maroon']

    portrayal = {
        'Shape': 'circle',
        'Layer': 0,
        'Cell id': str(agent.unique_id),
        'r': 0.25,
        'Filled':True,
        'Cell Type':str(agent.cluster)}
    
    token = agent.model.tokenizer[str(agent.cluster)]
    
    #tt = {'Tumor':0, 'Stromal Normal':1,'T cell':2, 'Adipocytes':3, 'TRAC+ Cells':4,
     #'Plasma Cells':5,'B Cells':6, 'Plasmacytoid Dendritic':7,'Tumor Associated Stromal':8,
     #'Endothelial':9,'Myoepithelial':10,'Epithelial':11,'Macrophage':12,'Transitional Cells':13,
     #'Mast Cells':14}
    
    #tt2 = tt[str(agent.cluster)]
     
    if token < len(colours):
        portrayal['Color'] = colours[token]
        
    if agent.model.curr_step == 0:
        step = 1
    else:
        step = agent.model.curr_step
    
    if agent.iqr == 'very_low':
        portrayal['r'] = 0.16
    elif agent.iqr == 'low':
        portrayal['r'] = 0.33
    elif agent.iqr == 'med_low':
        portrayal['r'] = 0.50
    elif agent.iqr == 'med_high':
        portrayal['r'] = 0.66
    elif agent.iqr == 'high':
        portrayal['r'] = 0.83
    else:
        portrayal['r'] = 1.0
    
    portrayal['Receiving_score'] = agent.num_r / step
    return portrayal



def visualization(adata, model_params, dist_on = True, port = 8521):
    #Colour cell based on cluster
    colours = ['violet', 'brown', 'pink','red','green', 'white', 'orange', 'grey', 'teal', 'navy', 'blue', 'magenta', 'cyan', 'yellow','maroon']
    
    model = CellModel
    dist_on = False
    if dist_on:
        grid2 = CanvasGrid(agent_portrayal, 75,55, 700, 700)
    else: 
        grid2 = CanvasGrid(agent_portrayal, 75,55, 700, 700)
    
    server = ModularServer(model,
                           [grid2],
                           'CellAgentChat',
                           model_params)
    server.port = port
    server.launch()

    

    
    
    

