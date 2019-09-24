"""
G2DR Skipgram Architecture based on Graph2Vec (Narayanan 2017)
"""

import tensorflow as tf
import os
import math
import logging

from corpus import Corpus
from time import time

# IF USING GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Skipgram(object):
	"""
	A Skipgram inspired architecture for distributed representations of graphs. 
	Refer to original Le and Mikolov Paper (2014)
	"""
	def __init__(self, num_graphs, num_subgraphs, learning_rate, embedding_size, num_negsample, num_steps, corpus):
		# super(Skipgram, self).__init__()
		self.num_graphs = num_graphs
		self.num_subgraphs = num_subgraphs
		self.learning_rate = learning_rate
		self.embedding_size = embedding_size
		self.num_negsample = num_negsample
		self.num_steps = num_steps
		self.corpus = corpus
		self.graph, self.batch_inputs, self.batch_labels, self.normalized_embeddings, self.loss, self.optimizer, self.graph_embeddings = self.trainer_initial()


	def trainer_initial(self):
		"""
		Sets the dataflow graph for the training phase
		"""
		graph = tf.Graph() # the computational tensorflow graph not one of our data instances.
		with graph.as_default():
			batch_inputs = tf.placeholder(tf.int32, shape=([None,]))
			batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))

			graph_embeddings = tf.Variable(
				tf.random_uniform([self.num_graphs, self.embedding_size], -0.5/self.embedding_size, 0.5/self.embedding_size))
			# graph_embeddings = tf.Variable(
			# 	tf.random_uniform([self.num_graphs, self.embedding_size], 0, 1))

			batch_graph_embeddings = tf.nn.embedding_lookup(graph_embeddings, batch_inputs) # hidden layer

			weights = tf.Variable(tf.truncated_normal([self.num_subgraphs, self.embedding_size],
														stddev=1.0/math.sqrt(self.embedding_size)))
			biases = tf.Variable(tf.zeros(self.num_subgraphs))

			# negative sampling
			loss = tf.reduce_mean(tf.nn.nce_loss(
				weights = weights, 
				biases = biases, 
				labels = batch_labels, 
				inputs = batch_graph_embeddings,
				num_sampled = self.num_negsample,
				num_classes = self.num_subgraphs,
				sampled_values = tf.nn.fixed_unigram_candidate_sampler(
					true_classes = batch_labels,
					num_true = 1,
					num_sampled=self.num_negsample,
					unique=True,
					range_max = self.num_subgraphs,
					distortion = 0.75,
					unigrams = self.corpus.subgraph_id_freq_map_as_list)
				)) # frequency of each word in dictionary in order by numerical id we gave it

			global_step = tf.Variable(0, trainable = False)  # the number of steps we have performed, we make sure not to change this.
			learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100000, 0.96, staircase=True) # linear decay, we can make this a bit more exciting
			learning_rate = tf.maximum(learning_rate, 0.001) # make sure the learning rate cannot go below 0.001 to ensure at least a minimal learning

			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

			norm = tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims = True))
			normalized_embeddings = graph_embeddings/norm

		return graph, batch_inputs, batch_labels, normalized_embeddings, loss, optimizer, graph_embeddings

	def train(self, corpus, batch_size):
		"""
		Trains the dataflow graph set in trainer_initial over the epochs using the supplied batch size
		"""
		with tf.Session(graph=self.graph, config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)) as sess:
			init = tf.global_variables_initializer()
			sess.run(init)

			loss = 0

			for i in range(self.num_steps):
				t0 = time()
				step = 0
				while corpus.epoch_flag == False:
					batch_data, batch_labels = corpus.generate_batch_from_file(batch_size) # get the (target, context) wordID pairs

					feed_dict = {self.batch_inputs:batch_data, self.batch_labels:batch_labels}
					_, loss_val = sess.run([self.optimizer, self.loss], feed_dict = feed_dict)
					loss += loss_val

					# if step % 100 == 0:
					# 	if step > 0:
					# 		average_loss = loss/step
					# 		logging.info( 'Epoch: %d : Average loss for step: %d : %f'%(i,step,average_loss))
					step += 1

				corpus.epoch_flag = False
				epoch_time = time() - t0
				logging.info('##########   Epoch: %d :  %f, %.2f sec.  ##########' % (i, loss/step,epoch_time))
				loss = 0

			final_embeddings = self.normalized_embeddings.eval()
			self.graph_embeddings_for_normals = self.graph_embeddings.eval()
			self.normalized_embeddings_for_normals = final_embeddings
		return final_embeddings

	def infer_vector(self, graph_wl_path, batch_size=256, epochs=10):
		"""
		Given a graph_document file, we infer its embedding based on the techniques in Le and Mikolov 2014, 

		We will assume that this truly is an unseen example and not one already considered in the corpus

		:param: graph_wl_path: a .g2v<depth> graph document
		:param: batch_size = 256: the batch size for the training
		:param: epochs=5: number of steps to train the network to get the inferred vector 5 is the default based on the paper
		"""

		# make a new corpus with the supplied file
		newCorpus = Corpus(corpus_dir = self.corpus.corpus_dir, extension = self.corpus.extension)
		newCorpus.scan_and_load_corpus()
		newCorpus.add_file(graph_wl_path)
		new_graph_id = newCorpus._graph_name_to_id_map[graph_wl_path]


		# infer_graph = tf.Graph()
		# with infer_graph.as_default():
		batch_inputs = tf.placeholder(tf.int32, shape=([None,]))
		batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))

		old_graph_embeddings = tf.constant(self.graph_embeddings_for_normals) # the trained embeddings from earlier
		new_embedding = tf.Variable(tf.random_uniform([1, self.embedding_size], -0.5/self.embedding_size, 0.5/self.embedding_size))
		inf_graph_embeddings = tf.concat([old_graph_embeddings,new_embedding], 0)

		batch_graph_embeddings = tf.nn.embedding_lookup(inf_graph_embeddings, batch_inputs) # hidden layer

		weights = tf.Variable(tf.truncated_normal([newCorpus.num_subgraphs, self.embedding_size],
													stddev=1.0/math.sqrt(self.embedding_size)))
		biases = tf.Variable(tf.zeros(newCorpus.num_subgraphs))

		# negative sampling
		inf_loss = tf.reduce_mean(tf.nn.nce_loss(
			weights = weights, 
			biases = biases, 
			labels = batch_labels, 
			inputs = batch_graph_embeddings,
			num_sampled = self.num_negsample,
			num_classes = newCorpus.num_subgraphs,
			sampled_values = tf.nn.fixed_unigram_candidate_sampler(
				true_classes = batch_labels,
				num_true = 1,
				num_sampled=self.num_negsample,
				unique=True,
				range_max = newCorpus.num_subgraphs,
				distortion = 0.75,
				unigrams = newCorpus.subgraph_id_freq_map_as_list)
				# unigrams = self.corpus.subgraph_id_freq_map_as_list) # original
			))

		global_step = tf.Variable(0, trainable = False)  # the number of steps we have performed, we make sure not to change this.
		learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100000, 0.96, staircase=True) # linear decay, we can make this a bit more exciting
		learning_rate = tf.maximum(learning_rate, 0.001) # make sure the learning rate cannot go below 0.001 to ensure at least a minimal learning

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(inf_loss, global_step=global_step)

		norm = tf.sqrt(tf.reduce_mean(tf.square(inf_graph_embeddings), 1, keep_dims = True))
		inf_normalized_embeddings = inf_graph_embeddings/norm

		# use the new corpus to train the network for a bit and spit out the embeddings
		with tf.Session(config = tf.ConfigProto(allow_soft_placement=False)) as inf_sess:
			init = tf.global_variables_initializer()
			inf_sess.run(init)

			loss = 0
			for i in range(epochs):
				while newCorpus.epoch_flag == False:
					batch_data, batch_labelss = newCorpus.generate_batch_from_file(batch_size) # get the (target, context) wordID pairs

					feed_dict = {batch_inputs:batch_data, batch_labels:batch_labelss}
					_, loss_val = inf_sess.run([optimizer, inf_loss], feed_dict = feed_dict)
					loss += loss_val

				newCorpus.epoch_flag = False
				loss = 0

			final_embeddings = inf_normalized_embeddings.eval()
		return final_embeddings[new_graph_id]

	def infer_initial(self, newCorpus):
		"""
		Defines the graph to use for inference tasks
		"""
		infer_graph = tf.Graph()
		with infer_graph.as_default():
			batch_inputs = tf.placeholder(tf.int32, shape=([None,]))
			batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))

			old_graph_embeddings = tf.constant(self.graph_embeddings_for_normals) # the trained embeddings from earlier
			new_embedding = tf.Variable(tf.random_uniform([1, self.embedding_size], -0.5/self.embedding_size, 0.5/self.embedding_size))
			inf_graph_embeddings = tf.concat([old_graph_embeddings,new_embedding], 0)

			batch_graph_embeddings = tf.nn.embedding_lookup(inf_graph_embeddings, batch_inputs) # hidden layer

			weights = tf.Variable(tf.truncated_normal([newCorpus.num_subgraphs, self.embedding_size],
														stddev=1.0/math.sqrt(self.embedding_size)))
			biases = tf.Variable(tf.zeros(newCorpus.num_subgraphs))

			# negative sampling
			inf_loss = tf.reduce_mean(tf.nn.nce_loss(
				weights = weights, 
				biases = biases, 
				labels = batch_labels, 
				inputs = batch_graph_embeddings,
				num_sampled = self.num_negsample,
				num_classes = newCorpus.num_subgraphs,
				sampled_values = tf.nn.fixed_unigram_candidate_sampler(
					true_classes = batch_labels,
					num_true = 1,
					num_sampled=self.num_negsample,
					unique=True,
					range_max = newCorpus.num_subgraphs,
					distortion = 0.75,
					unigrams = newCorpus.subgraph_id_freq_map_as_list)
					# unigrams = self.corpus.subgraph_id_freq_map_as_list) # original
				))

			global_step = tf.Variable(0, trainable = False)  # the number of steps we have performed, we make sure not to change this.
			learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 100000, 0.96, staircase=True) # linear decay, we can make this a bit more exciting
			learning_rate = tf.maximum(learning_rate, 0.001) # make sure the learning rate cannot go below 0.001 to ensure at least a minimal learning

			optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(inf_loss, global_step=global_step)

			norm = tf.sqrt(tf.reduce_mean(tf.square(inf_graph_embeddings), 1, keep_dims = True))
			inf_normalized_embeddings = inf_graph_embeddings/norm

		return infer_graph, batch_inputs, batch_labels, inf_normalized_embeddings, inf_loss, optimizer, inf_graph_embeddings

	def new_infer_vector(self, graph_wl_path, batch_size=1024, epochs=10):
		"""
		Slightly new version faster version

		:param: graph_wl_path: a .g2v<depth> graph document
		:param: batch_size = 256: the batch size for the training
		:param: epochs=5: number of steps to train the network to get the inferred vector 5 is the default based on the paper
		"""

		# make a new corpus with the supplied file
		newCorpus = Corpus(corpus_dir = self.corpus.corpus_dir, extension = self.corpus.extension)
		newCorpus.scan_and_load_corpus()
		newCorpus.add_file(graph_wl_path)
		new_graph_id = newCorpus._graph_name_to_id_map[graph_wl_path]

		infer_graph, batch_inputs, batch_labels, inf_normalized_embeddings, inf_loss, optimizer, inf_graph_embeddings = self.infer_initial(newCorpus)

		# use the new corpus to train the network for a bit and spit out the embeddings
		with tf.Session(graph = infer_graph, config = tf.ConfigProto(allow_soft_placement=False)) as inf_sess:
			init = tf.global_variables_initializer()
			inf_sess.run(init)

			loss = 0
			for i in range(epochs):
				while newCorpus.epoch_flag == False:
					batch_data, batch_labelss = newCorpus.generate_batch_from_file(batch_size) # get the (target, context) wordID pairs

					feed_dict = {batch_inputs:batch_data, batch_labels:batch_labelss}
					_, loss_val = inf_sess.run([optimizer, inf_loss], feed_dict = feed_dict)
					loss += loss_val

				newCorpus.epoch_flag = False
				loss = 0

			final_embeddings = inf_normalized_embeddings.eval()
		return final_embeddings[new_graph_id]

# Basically complete we just have to add in the predict function
# manual testing code
if __name__ == '__main__':
	logger = logging.getLogger()
	logger.setLevel("INFO")

	corpus_dir = "data/mutag"
	embedding_size = 150
	epochs = 100
	learning_rate = 0.5
	wlk_h = 2
	output_dir = "embeddings"
	open_fname = "_".join([os.path.basename(corpus_dir), "dims", str(embedding_size), "epochs", str(epochs), "lr", str(learning_rate), "wlDepth", str(wlk_h), "embeddings.txt"])
	open_fname = os.path.join(output_dir, open_fname)
	extension = "g2v"+str(wlk_h)
	num_negsample = 10
	batch_size = 256

	logging.info("Initializing SKIPGRAM")
	corpus = Corpus(corpus_dir, extension, max_files=0)
	corpus.scan_and_load_corpus()

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

	# infer a vector
	model_skipgram.infer_vector("data/mutag/0.gexf.g2v2")