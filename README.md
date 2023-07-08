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
