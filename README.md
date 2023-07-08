# CellAgentChat

## Overview
CellAgentChat constitutes a comprehensive framework integrating gene expression data and existing knowledge of signaling ligand-receptor interactions to compute the probabilities of cell-cell communication. Utilizing the principles of agent-based modeling (ABM), we describe each cell agent through various attributes, including cell identification (e.g. cell type or clusters), gene expression profiles, ligand-receptor universe, spatial coordinates (optional), and pseudotime (optional). The evaluation of cellular communication between sender and receiver cells involves a triad of interconnected components: ligand diffusion rate (γ~l~), receptor receiving rate (α~r~), and receptor conversion rate (β~r~).

The initial step in our approach involves the computation of the ligand diffusion rate (γ~l~), which quantifies the ligands secreted by the sender cell that reach the target receiver cell. This rate is a function of both ligand expression and intercellular distance. In the subsequent step, CellAgentChat estimates the receptor receiving rate (α~r~), which signifies the fraction of secreted ligands received by the receptor on the receiver cell. We then leverage a neural network to evaluate the receptor conversion rate (β~r~), which measures the potency of a cellular interaction by studying its effect on downstream gene expression dynamics. Considering the possibility that a similar interaction strength between ligand-receptor pairs can yield diverse impacts on downstream gene expression dynamics, it becomes crucial to assess the conversion rate for each receptor. The interaction between a specific ligand-receptor pair from sender and receiver cell agents is quantified by ligand-receptor scores (LR scores), which depend on the aforementioned rates. A ligand-receptor pair is deemed significant if its LR score significantly surpasses the background score derived from a permutation test. The cumulative interaction between two cell clusters is then computed as the total number of significant ligand-receptor pairs. 

The ABM framework further allows us to adjust the agent behavior rules, facilitating the exploration of long and short-range cellular interactions and the effects of in-silico receptor blocking. This approach also supports the agent-based animation and visualization of cellular interactions associated with individual cells within the biological system, enhancing the interpretability and visualization of the modeling outcomes.

<img width="1000" alt="image" src="https://github.com/mcgilldinglab/CellAgentChat/assets/77021753/5301e340-202b-4320-bf94-f46122ee8a07">

## Key Capabilities

1. Estimating cell-cell interactions by utilizing spatial coordinates derived from spatial transcriptomics data.
2. Determining cell-cell interactions by analyzing the pseudotime trajectory of cells.
Inferring cell communication between different cell type populations and identifying relevant ligand-receptor pairs.
3. Assessing the receiving strength of individual cells through the utilization of our animation platform and agent-based modeling techniques.
4. Efficiently tracking specific ligand-receptor pairs, cells or cell types of interest on our animation platform for detailed analysis.
5. Evaluating the impact of cell-cell interactions on downstream genes.
6. Performing effective in-silico perturbations, such as manipulating receptor blockage, to identify target genes for novel therapeutic approaches.
7. Inferring both short and long-range interactions between cells.

## Installation

### Prerequisites

* Python >= 3.10
* R >= 4.2.2
* Python dependencies
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

This is a three column csv or text file that contaisn the ligand receptor pairs. We provide a ligand-receptor database from CellTalkDB. However, the user can opt to provide their own custom database. The first column contains the **lr_pair**, the second column contains the **ligand_gene_symbol** and the third column contains the **receptor_gene_symbol**. Note: the gene/protein ids must be the gene names, matching the gene expression file. All other columns will be ignored. 

#### Gene Expression file (Mandatory)

This is a cell X genes csv or text file. This file can contain the unnormalized counts (we provide a preprocessing function to normalize the counts) for each gene or normalized values. Note: the gene/protein ids must be the gene names. 

#### Meta file (Mandatory)

This is a two column csv or text file indicating the cell type/cluster name that each cell resides in. The first column should be the **cell** and the second column should be the **cell_type**. 

#### Spatial Coordinates file (Optional)

This is a three column csv or text file that contains the spatial coordinates of each cell determined from spatial transcriptomics data. The first column should be the **cell** (matching those in **meta.txt**). The second and third columns should be the **x** and **y** coordinates, respectively. 

#### Pseudotime file (Optional)

This is a three column csv or text file that contains the pseudotime values of each cell. The first column should be the **cell** (matching those in **meta.txt**) and the second column should be the pseudotime values of each cell. Alternatively, we provide the option to calculate the pseudotime values manually using Slingshot.

#### Anndata object (h5ad file) Optional)

The gene expression, cell type, spatial coordinates and pseudotime values can also all be stored in an anndata input file. The gene expression should be stored in the data matrix **X**. The cell types and spatial coordinates (optional) and pseudotime (optional) should be stored in the observations under the column names **cell_type**, **x**, **y** and **pseudotime**, respectively. 

