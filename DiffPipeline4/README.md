# Pipeline 4 : Experiment Area

## Basics 

AgePredict() is the original age prediction model

The basic (pre-"aligned data") dataset is denoted as BTRN, BTST
The basic "aligned data" dataset is denoted as BATRN, BATST
APREF = (Q,R) pairs where Q is from BATST, R from BATRN such that age(R) = AgePredict(Q)

## Architecture

Trained on [Q from BATRN, R from BATRN]
Testing on [Q from BATST, R from BATST]
Testing on APREF

Using ordinal regression

## Source

diff_pipeline4_dataset* - several kinds of the dataset module. Noteable ones:
- diff_pipeline4__dataset_simple.py - selection of pairs according to radius range and embeddings nearset neighbours
- diff_pipeline4__dataset_rich - more generalized options with configuration flags on top of the file. In addition to the radius range and embeddings nearset neighbours, also selection per age according to relevant references.

- train_diff_ordinal.ipynb - USE ONLY THIS FOR RUNNING TRAINING (TODO : need to update diff_pipeline4__train.py probably)

- infer_diff_ordinal.ipynb - run inference for diff only

- infer_age_with_fix_ordinal.ipynb - running end-to-end inference using 2 models (AgePredict() + diff)

- age_predict_embeddings - extracting embeddings from AgePredict (Deep Age initial model for this project - the one we try to optimize)
TODO: change to use torchextractor (easier and faster)

- analyze_age_predict_vis - visualizations for embeddings + saving map files to arrays

- analyze_diff_model - embeddings analysis of diff model



## Research

A lot of research done here:
- diff prediction, radius range
- diff prediction, ranges - 0-4, 4-10 but the treatment with ordinal regression can be done better way (we used here same model to train -10-(-4) and 4-10 ranges diff detection which seems less correct to do)
- embeddings research: the face2emb files are embeddings extracted for the BATRN and BATST sets

Recommendation: not stay here but create new pipelines

