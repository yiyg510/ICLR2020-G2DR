"""
Utility functions to help train neural language models
"""

import os
import logging
from corpus import Corpus
from skipgram import Skipgram
from utils import save_graph_embeddings

def train_skipgram(corpus_dir, extension, learning_rate, embedding_size, 
    num_negsample, epochs, batch_size, wlk_h ,output_dir, min_count=0):
    """
    Trains the skipgram model for the corpus of graphs and returns the embeddings
    we still need to return the model instead

    :param: corpus_dir: directory with WL kernel relabeled files. All the files in this folder will be relabeled
                        according to the WL relabeling strategy and the format of each line of this directory will be:
                        <target> <context 1> <context 2>....
    :param: extension: extension of the relabeled file typically (default)) ".gexf.g2v3"
    :param: learning_rate: learning rate for the skipgram, currently also involves a linear decay (TODO:add adam or rmsprop)
    :param: embedding_size: number of dimensions the graph embeddings should have
    :param: num_negsample: number of negative samples to be used by the skipgram model
    :param: epochs: number of iterations the entire dataset is traversed by the skipgram model
    :param: batch_size: the size of each batch fed into the skipgram model
    :param: wlksize: wlk kernel depth
    :param: output_dir: the folder where the embedding file will be saved
    :return: name of the file that contains the embeddings in json format
    """
    
    # Create final embedding file_name and check if embeddings have already been made or not
    open_fname = "_".join([os.path.basename(corpus_dir), "dims", str(embedding_size), "epochs", str(epochs), "lr", str(learning_rate), "wlDepth", str(wlk_h), "embeddings.txt"])
    open_fname = os.path.join(output_dir, open_fname)
    if os.path.isfile(open_fname):
        logging.info("The embedding file: %s is already there, we dont need to train the skipgram" % (open_fname))
        return open_fname

    # Initialize the skipgram by loading the corpus object
    logging.info("Initializing SKIPGRAM")
    corpus = Corpus(corpus_dir, extension, max_files=0, min_count=min_count)
    corpus.scan_and_load_corpus()
    logging.info("Number of subgraphs: %s" % (corpus.num_subgraphs))

    model_skipgram = Skipgram(
        num_graphs = corpus.num_graphs,
        num_subgraphs = corpus.num_subgraphs,
        learning_rate = learning_rate,
        embedding_size = embedding_size,
        num_negsample = num_negsample,
        num_steps = epochs,
        corpus = corpus
    )
    logging.info("Training SKIPGRAM")
    final_embeddings = model_skipgram.train(corpus = corpus, batch_size = batch_size)

    logging.info("Write the matrix to a word2vec json file")
    save_graph_embeddings(corpus, final_embeddings, open_fname)
    
    logging.info("Completed writing the final embeddings, this can be found in file %s" % (open_fname))
    return open_fname

# In line with original (Complete)