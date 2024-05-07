# CellAgentChat

## Overview
CellAgentChat constitutes a comprehensive framework integrating gene expression data and existing knowledge of signaling ligand-receptor interactions to compute the probabilities of cell-cell communication. Utilizing the principles of agent-based modeling (ABM), we characterize each cell agent through various attributes, including cell identities (e.g. cell type or clusters), gene expression profiles, ligand-receptor universe and spatial coordinates (optional). We quantify cellular interactions between sender and receiver cells based on the number of ligands secreted by the sender cells and subsequently received by the receiver cells. This process hinges upon three interrelated components: ligand diffusion rate (γ<sup>l</sup>), receptor receiving rate (α<sup>r</sup>), and receptor conversion rate (β<sup>r</sup>).

The initial step in our approach involves the estimation of the receptor receiving rate (α<sup>r</sup>), which signifies the fraction of secreted ligands received by the receptor on the receiver cell. We then leverage a neural network to evaluate the receptor conversion rate (β<sup>r</sup>), which measures the potency of a cellular interaction by studying its effect on downstream gene expression dynamics. Considering the possibility that a similar interaction strength between ligand-receptor pairs can yield diverse impacts on downstream gene expression dynamics, it becomes crucial to assess the conversion rate for each receptor. Next, our approach involves the computation of the ligand diffusion rate (γ<sup>l</sup> = _x_<sup>l</sup>/_d_<sup>τ×δ</sup>), which quantifies the ligands secreted by the sender cell that reach the target receiver cell. This rate is a function of both ligand expression, _x_<sup>l</sup> and intercellular distance, _d_. The ligand diffusion rate also governed by two parameters: τ and δ. τ represents the degrees of spatial freedom concerning the single-cell data, typically set to two for spatial transcriptomics data that is derived from two-dimensional slices. δ signifies the decay rate of ligand diffusion which prioritizes interactions over long or short distances. When δ is lower than 1, it weakens the decay rate, giving precedence to long-range interactions. Conversely, when δ exceeds 1, it amplifies the decay rate, favoring short-range interactions. A δ = 1 (default) equally prioritizes long and short-range interactions (no preference). 

We quantify the interaction between a specific LR pair from sender and receiver cell agents using LR scores, which depend on the above rates. The LR score for two cluster pairs is the Interaction Score (IS). We consider an IS for an LR pair significant if its IS score significantly surpasses the background score derived from a permutation test. The cumulative interaction between two cell clusters is computed as the total number of significant LR pairs. CellAgentChat also computes the cell receiving score (CRS) to measure the received interactions for each individual cell.

The ABM framework further allows us to adjust the agent behavior rules, facilitating the exploration of long and short-range cellular interactions analyzing the effects of in-silico receptor blocking and studying the temporal dynamics of genes through dynamic simulations. This approach also supports the agent-based animation and visualization of cellular interactions associated with individual cells within the biological system, enhancing the interpretability and visualization of the modeling outcomes.

![Figure 1](https://github.com/mcgilldinglab/CellAgentChat/assets/77021753/083bb697-c7ca-4609-ba9c-63d3ba2a31a1)

## Key Capabilities

1. Estimating cell-cell interactions by utilizing spatial coordinates derived from spatial transcriptomics data.
2. Determining cell-cell interactions by analyzing the pseudotime trajectory of cells.
3. Inferring cell communication between different cell type populations and identifying relevant ligand-receptor pairs.
4. Assessing the receiving strength of individual cells through the utilization of our animation platform and agent-based modeling techniques.
5. Efficiently tracking specific ligand-receptor pairs, cells or cell types of interest on our animation platform for detailed analysis.
6. Evaluating the impact of cell-cell interactions on downstream genes.
7. Performing effective in-silico perturbations, such as manipulating receptor blockage, to identify target genes for novel therapeutic approaches.
8. Inferring both short and long-range interactions between cells.
9. Analyze the impact of cell-cell interactions on cell states over short time duration through dynamic ABM simulations. 

## Installation

### Prerequisites

* Python >= 3.10
* R >= 4.2.2
* Python dependencies
    * numpy >= 1.22.3
    * pandas >= 1.5.0
    * scanpy >= 1.9.1
    * Mesa >= 1.0.0
    * torch >= 1.12.1
    * scipy >= 1.9.1
    * seaborn >= 0.12.0
    * matplotlib >= 3.6.0
    * pyslingshot >= 0.0.2 (https://github.com/mossjacob/pyslingshot)
* R dependencies
    * tidyverse >= 2.0.0
    * ComplexHeatmap >= 2.14.0
    * BiocManager >= 1.30.19
    * reshape >= 0.8.9
    * optparse >= 1.7.3
    * utils >= 4.2.2

### Installing CellAgentChat

You may install CellAgentChat and its dependencies by the following command:

```
pip3 install git+https://github.com/mcgilldinglab/CellAgentChat
```

## Input Preparation

#### Ligand-Receptor Database (Mandatory)

This is a three column csv, tsv or text file that contains the ligand receptor pairs. We provide a ligand-receptor database from CellTalkDB (see human database [here](https://github.com/mcgilldinglab/CellAgentChat/blob/main/src/human_lr_pair.tsv)). However, the user can opt to provide their own custom database. The first column contains the ```lr_pair```, the second column contains the ```ligand_gene_symbol``` and the third column contains the ```receptor_gene_symbol```. Note: the gene/protein ids must be the gene names, matching the gene expression file. All other columns will be ignored. 

#### Gene Expression file (Mandatory)

This is a cell X genes csv or text file. This file can contain the unnormalized counts (we provide a preprocessing function to normalize the counts) for each gene or normalized values (see [example](https://github.com/mcgilldinglab/CellAgentChat/blob/main/tutorial/gene_expression.csv.zip)). Note: the gene/protein ids must be the gene names. 

#### Meta file (Mandatory)

This is a three column csv or text file indicating the cell type/cluster name and batch of each cell (see [example](https://github.com/mcgilldinglab/CellAgentChat/blob/main/tutorial/meta.csv)). The first column should be the ```cell```, the second column should be the ```cell_type``` and the third column should be the ```Batch```. 

#### Spatial Coordinates file (Optional)

This is a three column csv or text file that contains the spatial coordinates of each cell determined from spatial transcriptomics data (see [example](https://github.com/mcgilldinglab/CellAgentChat/blob/main/tutorial/spatial_coordinates.csv)). The first column should be the ```cell``` (matching those in ```meta.txt```). The second and third columns should be the ```x``` and ```y``` coordinates, respectively. 

#### Pseudotime file (Optional)

This is a two column csv or text file that contains the pseudotime values of each cell (see [example](https://github.com/mcgilldinglab/CellAgentChat/blob/main/tutorial/pseudotime.csv)). The first column should be the ```cell``` (matching those in ```meta.txt```) and the second column should be the pseudotime values of each cell. Alternatively, we provide the option to calculate the pseudotime values manually using Slingshot.

#### Anndata object (h5ad file) Optional)

The gene expression, cell type, spatial coordinates and pseudotime values can also all be stored in an anndata input file. The gene expression should be stored in the data matrix ```X```. The cell types and spatial coordinates (optional) and pseudotime (optional) should be stored in the observations under the column names ```cell_type```, ```x```, ```y``` and ```pseudotime```, respectively. 

## Tutorials

Please check the tutorial directory of the repository.

* [Full tutorial for CellAgentChat analysis of a single dataset WITHOUT pseudotime with explanation of each function and all additional features](https://htmlpreview.github.io/?https://github.com/mcgilldinglab/CellAgentChat/blob/main/tutorial/Tutorial.ipynb)

* Full tutorial for CellAgentChat analysis of a single dataset WITH pseudotime coming soon

## Animation

Our animation platform offers real-time visualization of the receiving interactions between individual cells. Additionally, users have the flexibility to perform effective in-silico perturbations by manipulating various parameters of the model.

To access the tutorial on server setup for our animation platform, please consult part 4 of either of our tutorials, available with or without pseudotime.

For a comprehensive tutorial on how to use our animation platform and effectively manipulate the available parameters, we highly recommend watching our tutorial video. It will provide you with a step-by-step guide on utilizing the platform to its fullest potential.

Tutorial Video: 
