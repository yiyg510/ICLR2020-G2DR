# How can we generalise learning distributed representations of graphs, G2DR instance implementation for replication.

![G2DR Overview](https://user-images.githubusercontent.com/3254542/65468317-ab5e8780-de5b-11e9-94d5-89d3fda4f93e.png)

This directory contains a anonymised implementation of G2DR which is an instance of the framework we describe for the ICLR 2020 Submission of "How can we generalise learning distributed representations of graphs?". Code has been specially commented for friendly exposition, and most modules contain test routines which show self-contained examples, as well as two scripts to run the whole process based on single or batch experiment runs.

This work is heavily inspired by Deep Graph Kernels (Yanardag and Vishwanathan, 2015) and Graph2Vec (Narayanan et al. 2017) motivated around combining their approaches into a single unified methodology, which could also be applied to other discrete structures beyond graphs under the R-Convolutional Framework (Haussler 1999). This is exemplified by G2DR as a derived instance of Graph2Vec extending beyond labelled graphs towards unlabelled graphs which are tested in our paper. This was motivated by the interest in applying the powerful Graph2Vec model beyond small labeled graphs towards larger ones as described in Yanardag et al.

For more detail please consult our paper submission.

# Prerequisites
G2DR is implemented in Python 3.6+ upon the Tensorflow framework and requires the following packages.

- Numpy
- NetworkX
- Scikit-Learn
- Pandas
- terminaltables
- tensorflow

These may be installed within a virtual environment running Python 3.6+ with pip

```bash
pip install numpy pandas networkx scikit-learn pandas terminaltables
```
and following instructions for installing tensorflow (CPU or GPU) we recommend the stable CPU package if you are just quickly testing the code.
```bash
pip install tensorflow 
```

# Usage
The `single_experiment.py`, which is to be run from inside the `src` directory, runs through the whole process of

1. Decomposing graphs into subgraphs (canonical forms)
2. Generating the graphdoc corpus
3. Learning embeddings using the neural language model
4. Performing a classification with trained embeddings through 10 Fold Cross Validation

With hyper-parameters/datasets/etc. that can be set by the user. Otherwise the `batch_experiment.py` file may be used to test a larger scope of hyper-parameters mainly on embedding size and subtree degree. Learned embeddings will be saved in "embeddings" folder specified by either script file. This may need to be generated by the user before hand. Results will be recorded in a CSV table as well as an ASCIItable in the terminal.

## Overview
Aside from the `single_experiment.py` and `batch_experiment.py` files the repository containes the following files:

- `classify.py`: Module containing various functions for classification (on top of the learned embeddings)
- `corpus.py`: Definition of corpus class which represents the corpus of graph documents
- `make_graphdoc_corpus.py`: Routines for reduction of graph datasets into subgraph patterns and graph documents
- `skipgram.py`: Python3.6 Tensorflow implementation of PV-DBOW model for G2DR with negative sampling 
- `train_utils.py`: routine to run and debug skipgram instances
- `utils.py`: general I/O utilities

In addition to the code files the `src` folder contains a few directories

- `embeddings`: contains trained embeddings of G2DR for downstream application
- `scores`: contains saved instances of experiments for record keeping as csv files
- `data`: contains the data as well as the graph documents generated in respective folders of datasets

## Data
Due to size constraints of the GitHub platform and portability we have included the MUTAG dataset. Additional datasets can be downloaded from Kersting et al. https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

## Attribution to Graph2Vec code
Narayanan et al's Python 2.7 implementation of Graph2Vec upon which our work is based can be found in https://github.com/MLDroid/graph2vec_tf.
