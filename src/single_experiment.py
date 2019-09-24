"""
Single Experiment

This script runs through the whole process of 
- decomposing graphs into subgraphs (canonical forms)
- generating the graphdoc corpus
- learning embeddings using the neural language model
- performing a classification with trained embeddings
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
corpus_dir = "data/mutag"
label_field_name = "Label"          
class_labels_fname = "data/mutag.Labels"
results_csv = "mutag_results.csv"
output_dir = "embeddings"
batch_size = 512
epochs = 1000
num_negsample = 10
learning_rate = 0.5
minCount = 1

# Graph-Doc Corpus Settings
wlk_h = 2
wl_extension = "g2v"+str(wlk_h)

# Skipgram Settings
embedding_size = 128

# Scores Settings
scores_dir = "scores"
csv_to_save = "_".join([str(batch_size), str(epochs), str(wlk_h), ".csv"])

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
# perform classification and evaluation
# classify_scores = perform_classification(corpus_dir, wl_extension, embedding_fname, class_labels_fname)
# acc, prec, recall, f_score = classify_scores
# data_input = [os.path.basename(corpus_dir), embedding_size, wlk_h, epochs, acc, prec, recall, f_score]
# data.append(data_input)

# to_print.append((os.path.basename(corpus_dir), embedding_size, wlk_h, epochs, acc))

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