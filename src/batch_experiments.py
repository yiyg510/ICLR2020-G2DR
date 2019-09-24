"""
Batch Experiments Script. 


This script runs through the whole process of 
- decomposing graphs into subgraphs (canonical forms)
- generating the graphdoc corpus
- learning embeddings using the graph neural language model
- performing classification with trained embeddings in 10 fold cross validation
"""

import os
import time
from time import time
from terminaltables import AsciiTable
import pandas as pd
import logging

# Our modules
from utils import get_files
from make_graphdoc_corpus import *
from train_utils import train_skipgram
from classify import perform_classification, cross_val_accuracy

## Debug
########
logger = logging.getLogger()
logger.setLevel("INFO")

############################################
################ Settings ##################
############################################
dataset = "mutag"
corpus_dir = "data/" + dataset
label_field_name = "Label"          
class_labels_fname = "data/mutag.Labels"
results_csv = "mutag_results.csv"
output_dir = "embeddings"
batch_size = 512
epochs = 1000
num_negsample = 10
learning_rate = 0.5
minCount = 1 # the minimum number of times a subgraph pattern has to occur to be included in the vocabulary

# Scores Settings
scores_dir = "scores"
csv_to_save = "_".join([dataset, str(learning_rate), str(minCount), ".csv"])

############################################

## Checks to make sure dataset and output paths exist
assert os.path.exists(corpus_dir), "File %s does not exist" % (corpus_dir)
assert os.path.exists(output_dir), "directory %s does not exist" % (output_dir)
graph_files = get_files(dname = corpus_dir, extension = ".gexf", max_files = 0)
print ("Loaded %s graph file names from %s" % (len(graph_files), corpus_dir))
# Pandas Dataframe header
header = ["corpus_dir", "embedding_size", "wlk_h", "epochs", "accuracy","precision","recall","fbeta"]
cv_header = ["corpus_dir", "embedding_size", "wlk_h", "epochs", "mean accuracy","standard deviation"]
data = []
# ASCIITABLE header
to_print = [("corpus_dir", "embedding_size", "wlk_h", "epochs", "accuracy")]
cv_to_print = [("corpus_dir", "embedding_size", "wlk_h", "epochs", "mean accuracy", "standard deviation")]


for wlk_h in range(2,5):
    wl_extension = "g2v"+str(wlk_h)
    for embedding_size in [8,16,32,64,128,256,512,1024]:
        #############################################
        #### Corpus Generation
        #############################################
        print("WL Subgraphs and Corpus Generation")
        t0 = time()
        wlk_relabeled_corpus(graph_files, max_h=wlk_h, node_label_attr_name=label_field_name)
        print("Generated Graph Document Corpus in %s seconds" % (round(time()-t0, 2)))

        #############################################
        #### Neural Language Model Training
        #############################################
        print ("SKIPGRAM LEARNING PHASE")
        # train the skipgram architecture
        t0 = time()
        embedding_fname = train_skipgram(corpus_dir, wl_extension, learning_rate, 
            embedding_size, num_negsample, epochs, batch_size, wlk_h,  output_dir, min_count=minCount)
        print ("Trained the skipgram model in %s seconds" % (round(time()-t0, 2)))    
        print ("SKIPGRAM LEARNING DONE")

        #############################################
        #### Classification Phase on Learned Embeddings
        #############################################
        print ("DOING classification")
        # perform single classification and evaluation
        # classify_scores = perform_classification(corpus_dir, wl_extension, embedding_fname, class_labels_fname)
        # acc, prec, recall, f_score = classify_scores
        # data_input = [os.path.basename(corpus_dir), embedding_size, wlk_h, epochs, acc, prec, recall, f_score]
        # data.append(data_input)

        # to_print.append((os.path.basename(corpus_dir), embedding_size, wlk_h, epochs, acc))
        # print ("Done classification")

        # 10 Fold Cross Validation with SVM
        classify_scores = cross_val_accuracy(corpus_dir, wl_extension, embedding_fname, class_labels_fname)
        mean_acc, std_dev = classify_scores
        data_input = [os.path.basename(corpus_dir), embedding_size, wlk_h, epochs, mean_acc, std_dev]
        data.append(data_input)

        cv_to_print.append((os.path.basename(corpus_dir), embedding_size, wlk_h, epochs, mean_acc, std_dev))
        print ("Done classification")


# Pandas
results_df = pd.DataFrame(data, columns=cv_header)
csv_to_save = scores_dir + "/" + csv_to_save
results_df.to_csv(csv_to_save)
# AsciiTable
print (AsciiTable(cv_to_print).table)