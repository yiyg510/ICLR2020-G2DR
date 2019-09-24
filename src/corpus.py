"""
Corpus class which represents the corpus of graph documents

"""

import numpy as np
from collections import defaultdict, Counter
from random import shuffle
from utils import get_files

class Corpus(object):
    """Class which represents all of the graph documents in a graph dataset"""
    def __init__(self, corpus_dir=None, extension=".g2v3", max_files=0, min_count=0):
        assert corpus_dir != None, "please specify the corpus folder"
        self.corpus_dir = corpus_dir
        self.extension = extension
        self.subgraph_index = 0
        self.graph_index = 0
        self.epoch_flag = 0
        self.max_files = max_files
        self.graph_ids_for_batch_traversal = []
        self.min_count = min_count

    def scan_and_load_corpus(self):
        """
        Gets the list of graph files, gives them number ids in a map and calls scan_corpus
        also makes available a list of shuffled graph ids for 
        """
        # Retrieve the graphs files and assign them internal ids for this method
        self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)
        self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)}
        self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}
            
        # Scan the corpus (ie explain in more detail in a sec)
        # This creates an "alphabet" and "vocabulary" of subgraphs
        subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

        self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
        shuffle(self.graph_ids_for_batch_traversal)

    def scan_corpus(self, min_count):
        """
        Maps the graph files to a subgraph alphabet from which we create new id's for the subgraphs
        which in turn get used by the skipgram architecture.
        """
        # Get all the subgraph labels (the centers, ie without context) first item each line
        subgraphs = []
        for fname in self.graph_fname_list:
            subgraphs.extend([l.split()[0] for l in open(fname).readlines()])
        subgraphs.append('UNK') # end flag

        # Frequency map of subgraph types (like WLSubgraph Kernel Vector Representations)
        subgraph_to_freq_map = Counter(subgraphs)
        del subgraphs

        # Remove infrequent graph patterns if user specified one.   
        if min_count:
            subgraph_to_freq_map = dict(subgraph_to_freq_map)
            for key in list(subgraph_to_freq_map.keys()):
                if subgraph_to_freq_map[key] < min_count:
                    subgraph_to_freq_map.pop(key, None)

        # Also give each of the subgraph labels even newer labels
        subgraph_to_id_map = {sg:i for i, sg, in enumerate(subgraph_to_freq_map.keys())}

        self._subgraph_to_freq_map = subgraph_to_freq_map
        self._subgraph_to_id_map = subgraph_to_id_map
        self._id_to_subgraph_map = {v:k for k,v in subgraph_to_id_map.items()}
        self._subgraphcount = sum(subgraph_to_freq_map.values()) #total num subgraphs in all graphs

        self.num_graphs = len(self.graph_fname_list) #doc size
        self.num_subgraphs = len(subgraph_to_id_map) #vocab size

        # This is a list sorted by id, which contains the frequency of a subgraph appearing as a list
        # Its usefulness is yet to reveal itself as does this implementation
        self.subgraph_id_freq_map_as_list = [] # id of this list is the subgraph/word id and value is the frequency of the subgraph
        for i in range(len(self._subgraph_to_freq_map)):
            self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])

        return self._subgraph_to_id_map

    def generate_batch_from_file(self, batch_size):
        """
        Generates batches for the skipgram using the corpus to train
        NB: "Context" here are in the sense of the graph as a whole
        so the centers ARE the context.

        :param batch_size: int size batch
        :return target_graph_ids: tuple of graphfname ids (mapped to a dict if need to find it
                context_word_outputs: a np vector of subgraphs, one from each graph in target_graph_ids

        """
        target_graph_ids = []
        context_subgraph_ids = []

        # Extract a random graph and read its contents
        graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]] # pull out a random graph (its filename here)
        graph_contents = open(graph_name).readlines()
        
        # tldr: random graph traverser 
        # If we've looked at all the subgraphs we go to the next graph
        # We set the epoch flag true if we've also gone through n graphs (in a n graph dataset)
        # we then grab the next graph file whether that be the next graph in the shuffled list
        # or the first graph in a reshuffled list of graph files
        while self.subgraph_index >= len(graph_contents):
            self.subgraph_index = 0
            self.graph_index += 1
            if self.graph_index == len(self.graph_fname_list):
                self.graph_index = 0
                np.random.shuffle(self.graph_ids_for_batch_traversal)
                self.epoch_flag = True
            graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
            graph_contents = open(graph_name).readlines()

        # Given that we haven't gotten enough graphs for our batch
        # We traverse the file at graph_name and graph the center as the (context subgraph)
        # which is a bit counter-intuitive but we consider the centers as the "context" of the
        # graph as a whole.

        while len(context_subgraph_ids) < batch_size:
            line_id = self.subgraph_index
            context_subgraph = graph_contents[line_id].split()[0] # first item on the line
            target_graph = graph_name

            if context_subgraph in self._subgraph_to_id_map:
                context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph]) # add the ids of the subgraph into the context
                target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target

            # context_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph]) # add the ids of the subgraph into the context
            # target_graph_ids.append(self._graph_name_to_id_map[target_graph]) # add the id of the graph for the target

            # move on to the next subgraph
            self.subgraph_index += 1

            # if we've reached the end of the graph file (ie exhasuted all the subgraphs in it)
            # we move onto the next graph as above.
            while self.subgraph_index == len(graph_contents):
                self.subgraph_index = 0
                self.graph_index +=1 
                if self.graph_index == len(self.graph_fname_list):
                    self.graph_index = 0
                    np.random.shuffle(self.graph_ids_for_batch_traversal)
                    self.epoch_flag = True

                graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
                graph_contents = open(graph_name).readlines()

        # Once we've built 'batch_size' number worth of targets and contexts
        # we zip them, shuffle them, and unzip the pairs in shuffled order (keeping the pairing)
        target_context_pairs = list(zip(target_graph_ids, context_subgraph_ids))
        shuffle(target_context_pairs)
        target_graph_ids, context_subgraph_ids = list(zip(*target_context_pairs))

        # make the shuffled (pair consistent) targets and contexts into np arrays
        target_context_pairs = np.array(target_graph_ids, dtype=np.int32)
        context_subgraph_ids = np.array(context_subgraph_ids, dtype=np.int32)

        # make a column of the context ids
        contextword_outputs = np.reshape(context_subgraph_ids, [len(context_subgraph_ids), 1])

        # returns the ids of the graph_files, and the context words
        return target_graph_ids, contextword_outputs

# technically complete as planned.
    
    def add_file(self, full_graph_path):
        """
        This function is for adding new graphs into corpus for inductive learning of new unseen graphs.
        """

        # Retrieve the graphs files and assign them internal ids for this method
        self.graph_fname_list = get_files(self.corpus_dir, self.extension, self.max_files)

        if full_graph_path in self.graph_fname_list:
            # already in the corpus so we ignore
            return
        else:
            self.graph_fname_list.append(full_graph_path) # and add the new file

            self._graph_name_to_id_map = {g:i for i, g in enumerate(self.graph_fname_list)} # add to the end
            self._graph_name_to_id_map[full_graph_path] = self.num_graphs
            self._id_to_graph_name_map = {i:g for g, i in self._graph_name_to_id_map.items()}
                
            # Scan the corpus (ie explain in more detail in a sec)
            # This creates an "alphabet" and "vocabulary" of subgraphs
            subgraph_to_id_map = self.scan_corpus(min_count=self.min_count)

            self.graph_ids_for_batch_traversal = list(range(self.num_graphs))
            shuffle(self.graph_ids_for_batch_traversal)


# for testing
if __name__ == '__main__':
    corpus_dir = "data/mutag"
    corpus = Corpus(corpus_dir, extension=".g2v2", max_files=100)

    corpus.scan_and_load_corpus()
    a,b = corpus.generate_batch_from_file(batch_size=50)

    additional_file = "data/mutag/99.gexf.g2v2"
    corpus.add_file(additional_file)