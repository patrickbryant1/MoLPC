# MoLPC
**M**odelling **o**f **L**arge **P**rotein **C**omplexes version 1.0

This directory contains a pipeline for predicting very large protein complexes using the
[FoldDock pipeline](https://gitlab.com/ElofssonLab/FoldDock) based on [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2).
\
\
[Here is the Colab notebook for MoLPC](https://colab.research.google.com/github/patrickbryant1/MoLPC/blob/master/MoLPC.ipynb)
\
\
AlphaFold2 is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) and so is FoldDock, which is a derivative thereof.  \
The AlphaFold2 parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode) and have not been modified.
\
**MoLPC** can be run using predictions of subcomponents from any method and is thus not directly dependent on AlphaFold2 - e.g. [AlphaFold-multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2) can also be used. Note that this results in approximately twice the run time. \
MolPC is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
\
\
**You may not use these files except in compliance with the licenses.**
\
\
Given a set of unique protein sequences and their **stoichiometry**,
this pipeline predicts the structure of an entire complex (without using any structural templates)
composed of the supplied sequences and stoichiometry. The pipeline is developed for protein complexes
with 10-30 chains, but is also functional for smaller protein complexes. \
\
Please see [Predicting the structure of large protein complexes using AlphaFold and Monte Carlo tree search](https://www.nature.com/articles/s41467-022-33729-4) for more information.

# The following sections contains information of how to install and run MoLPC locally

## Computational requirements
Before beginning the process of setting up this pipeline on your local system, make sure you have adequate computational resources.
The main bottleneck here is the structure prediction of trimeric subcomponents with AlphaFold2, which can require >40Gb of GPU RAM
depending on the number of residues in the subcomponent that is being predicted. Make sure you have available GPUs suitable for this
type of structure prediction as predicting with CPU will take an unreasonable amount of time. This pipeline assumes you have NVIDIA GPUs
on your system, readily available.


## Clone this repository
```
git clone https://github.com/patrickbryant1/MoLPC.git
```

## Install all packages and databases
```
bash setup.sh
```

## Predict
The following procedure outlines the steps taken to generate a protein complex prediction. \
\
This entire pipeline is present in the script pipeline.sh, which takes three
CSV files as input:
\
1: Unique sequences and their stoichiometry (labelled in numerical order) \
2: Chain sequences and their mapping to the unique sequences. E.g. if sequence 1
has stoichiometry 5, the chain sequences would be A,B,C,D,E. \
3: Optional - interactions between the chains (if known). E.g. A interacts with B,
B with F, F with G. \
\
A test case is provided in ./data/test/ for 1A8R, assuming no knowledge of interactions.
Example files are provided with suffixes **_useqs.csv**, **_chains.csv** and **_ints.csv** for
(1),(2) and (3). (3) is not used in this test case.
\
The script pipeline.sh takes these files as input to demonstrate the principle.
This script can be edited with your own input files to assemble new complexes.
\
To try this, simply do
```
bash pipeline.sh
```
\
The putput will be generated in **./data/test/**
\
**NOTE** \
If you are having singularity issues, please ensure singularity can access your CUDA drivers.
It may be necessary to bind the path of singularity to the file system you are running from.
Do this for the AlphaFold prediction step:
singularity exec --nv $SINGIMG python3 $BASE/src/AF2/run_alphafold.py
--> singularity exec --nv --bind PATH_TO_DATADIR:PATH_TO_DATADIR $SINGIMG python3 $BASE/src/AF2/run_alphafold.py
where the PATH_TO_DATADIR is the path to the base file system where you keep this code.
\
The pipeline consists of four steps:
## 1. MSA generation
Input: a fasta file containing sequences for each of the chains that are in your complex \
Output: MSAs that will be used to predict the structure of your complex
\
\
This protocol creates two MSAs constructed from a single search with [HHblits](https://toolkit.tuebingen.mpg.de/tools/hhblits) version 3.1.0 against [uniclust30_2018_08](https://academic.oup.com/nar/article/45/D1/D170/2605730?login=false). One MSA is paired using OX identifiers and one is block diagonalized.

## 2. Protein Folding
Input: MSAs and fasta files for each trimeric complex subcomponent according to step 1. \
Output: The predicted structure of each trimeric complex subcomponent

## 3. Assembly
From the interactions in the predicted subcomponents, we add chains sequentially following a predetermined path through the interaction network (graph). If two pairwise interactions are A-B and B-C, we assemble the complex A-B-C by superposing chain B from A-B and B-C using [BioPython’s SVD](https://biopython.org/docs/1.76/api/Bio.SVDSuperimposer.html) and rotating the missing chain to its correct relative position.

To find the optimal assembly route for a complex, we search for an optimal path using [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

## 4. Scoring - multiple interface predicted DockQ (mpDockQ)
After assembly, we score the interfaces of the complete complex using the average interface plDDT⋅log(number of interface contacts). These metrics are calculated for the entire interface of each chain, as in the [DockQ score](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879) for multiple interfaces. E.g. if chain A interacts with both chains B and C, the interface plDDT⋅log(number of interface contacts) is taken over both of these interfaces simultaneously. This is done for all interfaces and chains and averaged over the entire complex. The complexes with the highest scores are favored. A CSV file named "ID"_score.csv will be found in the output directory together with a pdb file of the complex. The scores in the CSV file can be used to calculate the mpDockQ accordingly:
\
\
*mpDockQ* = L/{1+exp(-k(x-x0))} + b ,
\
\
where x = average interface plDDT⋅log(number of interface contacts) averaged over all interfaces in the complex and L= 0.783, x0= 289.79, k= 0.061 and b= 0.23.

# Citation
If you use MolPC in your research, please cite
Bryant, P., Pozzati, G., Zhu, W. et al. Predicting the structure of large protein complexes using AlphaFold and Monte Carlo tree search. Nat Commun (2022). [link](https://www.nature.com/articles/s41467-022-33729-4)


--------------------------------------------------------------------------------------------
Copyright 2022 Patrick Bryant \
Licensed under the Apache License, Version 2.0. \
You may not use this file except in compliance with the License. \
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
