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
    'bins': Slider(
        'Max number of steps/Pseudotime Bins', 10, 5, 20, 1),
    'delta': Slider(
        'Delta', 1, 0, 10, 0.1),
    'shift': Slider(
        'Pseudotime Bin Shift', 0, -5, 5, 1),
    'tau': Slider(
        'Tau', 2, 1, 10, 1),
    'rec_block': Choice(
        'Block Receptor', value=False, choices= proteins), 
    'protein_choice': Choice(
        'Receptor Track', value=proteins[0], choices= proteins), 
    'lr_choice': Choice(
        'L-R Pair Track', value=pairs[0], choices= pairs),
    'dist': Checkbox(
        'Distance', value=True),
    'sender': Choice(
        'Sender', value=clusters[0], choices= clusters),
    'receiver': Choice(
        'Receiver', value=clusters[0], choices= clusters),
    }
    return model_params



def get_interactions(model):
    df1 = pd.DataFrame.from_dict(model.output4, orient='index').transpose()
    df1 = df1.rename({0:'Sum'})
    return df1
    
    
def get_lr_interactions(model):
    d1 = model.output5
    d2 = model.bins_used
    x = get_bins_used(d1,d2)
    df2 = pd.DataFrame.from_dict(x,orient='index').fillna(0).transpose()
    return df2


def get_lr_interactions2(model):
    d1 = model.output5
    d2 = model.bins_used
    x = get_bins_used(d1,d2)
    return x


def get_bins_used(d1, d2):
    cdict3 = {}
    d3 = {}
    for cpair in d1.keys():
        cdict = d1[cpair]
        cdict2 =d2[cpair]
        for lr in cdict.keys():
            cdict[lr] = cdict[lr] / cdict2[lr].count(True)
        d1[cpair] = cdict
    return d1


def get_avg_dist(model):
    return model.avg_dist


class CellAgent2(Agent):

    def __init__(self, unique_id, model, clust, exp, ptime, batch):
        super().__init__(unique_id, model)
        #Whether the cell is stationary or can move
        self.cluster = clust
        self.expression = exp
        #Visualization Tracking Fields
        self.num_r = 0
        self.ptime = ptime
        self.slice = batch
        self.iqr = "very_low"
        self.bin = None
        
        
    def Message(self):
        self.bin = self.model.curr_step
        #ligand expression for current bin
        bins = self.model.bins_dict
        cell_ligands = bins[self.ptime]
        self.num_r = 0
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
            """
            mean = np.average(unzipped_rec[1])
            std_dev = np.std(unzipped_rec[1])
            for i, val in enumerate(unzipped_rec[1]):
                #Add Receiving Rate to Rates
                if std_dev != 0:
                    rates2[unzipped_rec[0][i]] = self.model.rates[unzipped_rec[0][i]] * norm.cdf(val, loc=mean, scale=std_dev)
                else:
                    rates2[unzipped_rec[0][i]] = self.model.rates[unzipped_rec[0][i]]"""
        clust1 = self.cluster
        #Ligand receptor interactions
        
        if self.model.dist_param != 0:
            distances = self.model.distances
            c = distances[self.ptime][self]
            
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
            #counter
            if ckey in self.model.bins_used.keys():
                num_dict = self.model.bins_used[ckey]
            else:
                num_dict = {}
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
                            if self.model.delta != 1:
                                lig_delta = self.model.delta[lig]
                            else:
                                lig_delta = self.model.delta
                            new_dist2 = [1/(d**(self.model.dist_param*lig_delta))for d in dist2]
                            lig_score = np.multiply(exp, new_dist2).sum() / len(exp)  
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
                            
                            #Bin counter - make true for this step for lr pair
                            if lr_str in num_dict.keys():
                                bin_list = num_dict[lr_str]
                                bin_list[self.model.curr_step] = True
                                num_dict[lr_str] = bin_list
                            else:
                                tmp_list = [False]*self.model.max_steps
                                tmp_list[self.model.curr_step] = True
                                num_dict[lr_str] = tmp_list

            #Save interactions
            self.model.bins_used[ckey] = num_dict
            self.model.output[ckey] = cnum
            self.model.output2[ckey] = cdict
        #For Animation
        if self.model.receiver == clust1 or self.model.receiver == False:
            self.model.rec_score.append(self.num_r)


        
    def step(self):
        #Agent's step
        #Shifted pseudotime bin
        shifted_ptime = self.ptime + self.model.shift
        if shifted_ptime >= 0 and shifted_ptime <= self.model.max_steps and shifted_ptime == self.model.curr_step:
            #print("Hi, I am agent " + str(self.unique_id) + ", cluster is " + str(self.cluster))
            self.Message()
            
            

class CellModel2(Model):
    """
    CellAgentChat Parameters Help

    Number of Agents: The total count of cells utilized in the simulation.
    Max number of steps/Pseudotime bin: Number of bins to group cells of each cluster based on their pseudotime value. This value also represents the duration of the simulation in iterations.
    Delta: Influences the degree of cell-to-cell distance (default=1). For long-range mode, a delta value less than 1 is used, while for short-range mode, a delta value greater than 1 is employed.
    Pseudotime Bin Shift: A shift transformation applied to each pseudotime bin
    Tau: Represents the degree of freedom for distance (default=2).
    Block Receptor: Specifies the receptor to be obstructed.
    Receptor Track: Designates the receptor to be tracked by the animation. Only interactions involving the selected receptor will be displayed on the screen.
    L-R Pair: Identifies the L-R pair to be tracked by the animation. Only interactions involving the chosen L-R pair will be shown on the screen.
    Distance: Distance mode setting.
    Sender: Refers to the cell type responsible for sending ligands. Only interactions where the ligands originate from the specified sending cell type will be displayed on the screen.
    Receiver: Refers to the cell type responsible for receiving ligands. Only interactions where ligands are received by the specified receiving cell type will be shown on the screen.
    Gaussian Noise Percentage: The percentage of Gaussian noise added to ligand expression to enhance the dynamism of the simulations (default=5%).
    """
    
    
    def __init__(self, N, adata, lig_uni, rec_uni, rates, bins, dist, delta=1, shift=0, tau = 2,
                 rec_block=False, protein_choice=False, lr_choice=False,sender=False,receiver=False,
                 permutations=False, net = None):
        #Mandatory Inputs
        self.num_agents = N
        self.adata = adata
        self.lig_uni = lig_uni
        self.rec_uni = rec_uni
        self.rates = rates
        self.max_steps = bins
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
        self.protein_choice = protein_choice
        self.lr_choice = lr_choice
        self.dist_param = tau
        self.shift = shift
        self.rec_block = rec_block
        self.sender=sender
        self.receiver=receiver
        self.noise = noise
        
        #For Permutations
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
        
        self.output4 = {}
        self.output5 = {}
        
        #results
        self.results = {}
        self.results2 = {}
        
        #Helper Fields
        self.dist_pairs = {}
        self.ligs = 0
        self.distances = 0
        self.bins_dict = {}
        self.clusters = {}
        self.bins_count = []
        self.bins_used = {}
        self.rec_score = []
        self.tokenizer = {}
    
    
        #create grid
        if self.dist_on:
            self.grid = MultiGrid(max(adata.obs['x'])+1, max(adata.obs['y'])+1, True)
        else:
            self.grid = MultiGrid(51,51, True)
             
        #tokenizer
        cts = list(set(adata.obs['cell_type']))
        for i, clust in enumerate(cts):
            self.tokenizer[clust] = i
            
            
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
            #gene expression
            if scipy.sparse.issparse(adata.X):
                val = self.adata[adata.obs_names[i], self.genes].X.toarray()
                
            else:
                val = self.adata[adata.obs_names[i], self.genes].X
            val = list(val[0])
            exp = dict(zip(self.genes, val))
            
            #pseudotime
            ptime = adata.obs.loc[adata.obs_names[i], 'pseudotime']
                    
            a = CellAgent2(i, self, clust, exp, ptime, batch)
            self.schedule.add(a)
            
            #coordinates
            if self.dist_on:
                x, y = (adata.obs.loc[adata.obs_names[i], 'x'], adata.obs.loc[adata.obs_names[i], 'y'])
                self.grid.place_agent(a, (x, y))               
            else:
                x = random.randint(0,50)
                y = random.randint(0,50)
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
    
    
    def get_bins(self):
        new_clust = {}
        new_clust2 = {}
        for cell in self.schedule.agents:
            c = cell.cluster
            time = cell.ptime
            if c in new_clust.keys():
                new_clust[c].append(time)
                new_clust2[c].append(cell)
            else:
                new_clust[c] = [time]
                new_clust2[c] = [cell]
        count = 0 
        bins = [None]*self.max_steps
        bins2 = [0]*self.max_steps
        for c in new_clust.keys():
            times = new_clust[c]
            min_t = min(times)
            max_t = max(times)
            interval = (max_t - min_t)/self.max_steps
            int_list = []
            last = min_t
            for i in range(self.max_steps):
                last = last + interval
                int_list.append(last)
            for j, time1 in enumerate(times):
                done = False
                for k, val in enumerate(int_list):
                    if time1 <= val and not done:
                        count += 1
                        if type(bins[k]) == list:
                            alist = bins[k]
                            alist.append(new_clust2[c][j])
                            bins[k] = alist
                        else:
                            bins[k]= [new_clust2[c][j]]
                        new_clust2[c][j].ptime = k
                        bins2[k] += 1
                        done = True
                if time1 >= val and done == False:
                    count += 1
                    if type(bins[-1]) == list:
                            alist = bins[-1]
                            alist.append(new_clust2[c][j])
                            bins[-1] = alist
                    else:
                        bins[-1]= [new_clust2[c][j]]
                    new_clust2[c][j].ptime = self.max_steps-1
                    bins2[-1] += 1
                    done = True
        return bins   
    
    
    def get_bins2(self):
        #[{cluster1:[cell1,cell2,...], cluster2:[cell1,cell2,...]}, ...]
        bins = self.get_bins()
        bins2 = []
        bins3 = []
        for i in range(len(bins)):
            clust = {}
            clust3 = {}
            list1 = bins[i]
            for cell in list1:
                clust2 = cell.cluster
                if clust2 in clust.keys():
                    clust[clust2].append(cell)
                    clust3[clust2]+= 1
                else:
                    clust[clust2] = [cell]
                    clust3[clust2] = 1
            bins2.append(clust)
            bins3.append(clust3)
        self.bins_count = bins3
        return bins2   
            
        
    #include pseudotime
    def get_dist(self):
        #bins2 = [{cell:{clust:[distance1, distance2, ..]}}, ]
        bins = self.get_bins2()
        bins2 = [None]*self.max_steps
        new_dict = {}
        clusters = self.get_clusters()
        for cell in self.schedule.agents:
            adict = {}
            x = cell.ptime 
            bc = bins[x]
            for clust in bc.keys():
                cells = bc[clust]
                distances = []
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
            bins2[x] = new_dict
        return bins2
          
    
    #includes pseudotime
    def calc_ligands(self):
        #{cluster:{lig_name:[exp1,...,expn]}}
        #{time:{cluster:{lig_name:[exp1,...,expn]}}}
        bdict= {}
        bins = self.get_bins2()
        for i in range(len(bins)):
            clusters = bins[i]
            new_dict = {}
            for clust in clusters.keys():
                cells = clusters[clust]
                ligd = {}
                for lig in self.lig_uni.keys():
                    expl = []
                    for cell in cells:
                        exp = cell.expression[lig]
                        expl.append(exp)
                    if self.dist_param == 0:
                        expl = sum(expl) / len(expl)
                    ligd[lig] = expl
                new_dict[clust] = ligd
            bdict[i] = new_dict
        return new_dict, bdict
    
                
    def dist(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    
    def calc_dist(self):
        points = [cell.pos for cell in self.schedule.agents]
        distances = [self.dist(p1, p2) for p1, p2 in combinations(points, 2)]
        avg_distance = sum(distances) / len(distances)
        return avg_distance
        

    def step(self):
        """Advance the model by one step."""
        if (self.curr_step == 0):
            if self.dist_param != 0:
                self.avg_dist = self.calc_dist()
                print("Average Distance: "+str(self.avg_dist))
                self.distances = self.get_dist()
            self.ligs, self.bins_dict = self.calc_ligands()

        self.modify_db()
        self.schedule.step()
        
        #Save to results
        for cpair in self.output.keys():
            splitted = cpair.split('_')
            c2 = splitted[1]
            cdict = self.output2[cpair]
            #bin with shift
            index = self.curr_step-self.shift
            if index <= self.max_steps-1 and index > 0:
                num = self.bins_count[index][c2]
                if cpair in self.output5.keys():
                    cdict2 = self.output5[cpair]
                else:
                    cdict2 = {}
                for lr in cdict.keys():
                    val = sum(cdict[lr]) / num
                    if lr in cdict2.keys():
                        cdict2[lr] += val
                    else:
                        cdict2[lr] = val
                self.output5[cpair] = cdict2
                if cpair in self.output4.keys():
                    self.output4[cpair] += sum(list(self.output5[cpair].values()))
                else: 
                    self.output4[cpair] = sum(list(self.output5[cpair].values()))
        
        if not self.permutations:
            #Calculating ligand received IQR
            score2 = [x for x in self.rec_score]
            score2 = sorted(score2)
            if len(score2) != 0:
                b_size = len(score2) // 6
                high = b_size * 5
                med_high = b_size * 4
                med_low = b_size * 3
                low = b_size * 2
                very_low = b_size * 1
                for agent in self.schedule.agents:
                    received = agent.num_r
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
        
        self.output = {}
        self.output2 = {}
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



def agent_portrayal(agent):
    #Colour cell based on cluster
    colours = ['violet', 'brown', 'pink','red','green', 'orange', 'grey', 'teal', 'navy', 'blue', 'magenta', 'cyan', 'yellow','black', 'neon']

    
    portrayal = {
        'Shape': 'circle',
        'Layer': 0,
        'Cell id': str(agent.unique_id),
        'r': 0.25,
        'Filled':True,
        'Cell Type':str(agent.cluster),
        'Bin':str(agent.bin)}

    token = agent.model.tokenizer[str(agent.cluster)]
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
    
    portrayal['Receiving_score'] = agent.num_r
    return portrayal


#visualization
def visualization(adata, model_params, dist_on = True, port = 8521):
    #Colour cell based on cluster
    colours = ['violet', 'brown', 'pink','red','green', 'white', 'orange', 'grey', 'teal', 'navy', 'blue', 'magenta', 'cyan', 'yellow','black']
    
    model = CellModel2
    if dist_on:
        grid2 = CanvasGrid(agent_portrayal, max(adata.obs['x'])+1, max(adata.obs['y'])+1, 700, 700)
    else: 
        grid2 = CanvasGrid(agent_portrayal, 51,51, 700, 700)
    
    server = ModularServer(model,
                           [grid2],
                           'CellAgentChat',
                           model_params)
    server.port = port
    server.launch()
    

    
    






