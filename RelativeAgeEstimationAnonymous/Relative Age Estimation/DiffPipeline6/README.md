# Pipeline 6 : Radius Optimizer

## Basics 

The basic (pre-"aligned data") dataset is denoted as BTRN, BTST
The basic "aligned data" dataset is denoted as BATRN, BATST
APREF = (Q,R) pairs where Q is from BATST, R from BATRN such that age(R) = AgePredict(Q)


## Architecture

Trained on [Q from BATRN, R from BATRN]
Testing on [Q from BATST, R from BATST]
Testing on APREF

Using ordinal regression

Supporting multi-ref

NN arch: cnn + emb head + diff embeddings + maxpool + FC head

Configurable feature: using only nearest neighbours in embeddings space to select references

## Research

Currently best obtained results are a diff model doing age estimation in diff range 4-10.

## Nearest Neighbours

These are obtained either:
- original age model (AgePredict)
- recognition model

