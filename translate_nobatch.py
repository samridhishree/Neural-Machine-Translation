from collections import Counter, defaultdict
from itertools import count
import random
import _gdynet as dy
import numpy as np
import os
import sys
import nltk
from operator import itemgetter

dyparams = dy.DynetParams()
dyparams.set_mem(11000)
dyparams.init()

#Vocabulary related parameters
german_vocab_size = 0
english_vocab_size = 0
EOS = 0
BOS = 0
#Hyper-parameters of the model
lstm_num_of_layers = 2
embeddings_size = 512
state_size = 512
attention_size = 256

#Builds word-id indexer, word-to-int and int-to-word converter
class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

#Reads the sentences of the file and returns them as a list
def read(fname, pad_eos=True):
	fh = file(fname)
	for line in fh:
		if pad_eos == True:
			sent = "<s> " + line.strip() + " </s>"
		else:
			sent = line.strip()
		yield sent

german_train_file = sys.argv[1]
english_train_file = sys.argv[2]
model_file = sys.argv[3]

#Get the training sentences as a list
german_train = list(read(german_train_file))
english_train = list(read(english_train_file))
german_word_corpus = []
english_word_corpus = []

german_word_corpus.append("<UNK>")
english_word_corpus.append("<UNK>")

for sent in german_train:
	words = sent.split(" ")
	for word in words:
		german_word_corpus.append(word)

for sent in english_train:
	words = sent.split(" ")
	for word in words:
		english_word_corpus.append(word)

german_word_vocab = Vocab.from_corpus([german_word_corpus])
english_word_vocab = Vocab.from_corpus([english_word_corpus])
EOS = english_word_vocab.w2i["</s>"]
BOS = english_word_vocab.w2i["<s>"]

#Vocabulary Sizes
german_vocab_size = german_word_vocab.size()
english_vocab_size = english_word_vocab.size()

training_sentences = [(ger.strip().split(" "), eng.strip().split(" ")) for (ger, eng) in zip(german_train, english_train)]
indexed_train = []
for (german_list, english_list) in training_sentences:
	german_list = [german_word_vocab.w2i[word] for word in german_list]
	english_list = [english_word_vocab.w2i[word] for word in english_list]
	indexed_train.append((german_list, english_list))

#Declare and define the enc-doc models
model = dy.Model()
if(model_file == ""):
    enc_fwd_lstm = dy.LSTMBuilder(lstm_num_of_layers, embeddings_size, state_size, model)
    enc_bwd_lstm = dy.LSTMBuilder(lstm_num_of_layers, embeddings_size, state_size, model)
    dec_lstm = dy.LSTMBuilder(lstm_num_of_layers, ((state_size * 2) + embeddings_size), state_size, model)
    #Define the model parameters
    input_lookup = model.add_lookup_parameters((german_vocab_size, embeddings_size))
    attention_w1 = model.add_parameters(((attention_size, (state_size * 2))))
    attention_w2 = model.add_parameters(((attention_size, (state_size * lstm_num_of_layers * 2))))
    attention_v = model.add_parameters((1, attention_size))
    decoder_w = model.add_parameters((english_vocab_size, state_size + (state_size * 2)))
    decoder_b = model.add_parameters((english_vocab_size))
    output_lookup = model.add_lookup_parameters((english_vocab_size, embeddings_size))
else:
    print "Loading Model."
    [enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup, attention_w1, attention_w2, attention_v, decoder_w, decoder_b] = model.load(model_file)
    print "Model Loaded."

#Convert the input(german) sentence into its embedded form
def embed_sentence(sentence):
	global input_lookup
	return([input_lookup[w] for w in sentence])

#Run the given lstm model
def run_lstm(init_state, input_vecs):
	s = init_state
	out_vectors = []

	for vector in input_vecs:
		s = s.add_input(vector)
		out_vector = s.output()
		out_vectors.append(out_vector)

	return out_vectors

#Runs the bi-directional encoder for each word of the sentence passed.
#Returns a concatenated list of word-vectors (for each word in the sentence: [fwd_encoding, bwd_encoding])
def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
	sentence_rev = list(reversed(sentence))
	#Fwd-bwd encodings
	fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
	bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
	bwd_vectors = list(reversed(bwd_vectors))
	vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
	return vectors

#Return the attention score : MLP method
def attend(input_mat, state, w1dt):
	global attention_w2
	global attention_v
	w2 = dy.parameter(attention_w2)
	v = dy.parameter(attention_v)

	# input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
	# w1dt: (attdim x seqlen)
	# w2dt: (attdim x attdim)
	w2dt = w2*dy.concatenate(list(state.s()))
	# att_weights: (seqlen,) row vector
	unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
	att_weights = dy.softmax(unnormalized)
	# context: (encoder_state)
	context = input_mat * att_weights
	return context

#Decoder - given the encoded state for input sequence
def decode(dec_lstm, vectors, output):
	#Convert the words to word-ids
	w = dy.parameter(decoder_w)
	b = dy.parameter(decoder_b)
	w1 = dy.parameter(attention_w1)
	input_mat = dy.concatenate_cols(vectors)
	w1dt = None

	last_output_embeddings = output_lookup[output[-1]]
	#s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(state_size*2), last_output_embeddings]))
	#s = dec_lstm.initial_state([vectors[-1]])
	s = dec_lstm.initial_state()
	c_t_minus_1 = dy.vecInput(state_size*2)
	loss = []

	for word in output:
		w1dt = w1dt or w1 * input_mat
		vector = dy.concatenate([last_output_embeddings, c_t_minus_1])
		s = s.add_input(vector)
		h_t = s.output()
		c_t = attend(input_mat, s, w1dt)
		predicted = dy.affine_transform([b, w, dy.concatenate([h_t, c_t])])
		cur_loss = dy.pickneglogsoftmax(predicted, word)
		last_output_embeddings = output_lookup[word]
		loss.append(cur_loss)
		c_t_minus_1 = c_t

	loss = dy.esum(loss)
	#print "Loss = ", loss
	return loss

#Generate the translations from the current trained state of the model
def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
	embedded = embed_sentence(in_seq)
	encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

	w = dy.parameter(decoder_w)
	b = dy.parameter(decoder_b)
	w1 = dy.parameter(attention_w1)
	input_mat = dy.concatenate_cols(encoded)
	w1dt = None

	last_output_embeddings = output_lookup[BOS]
	#s = dec_lstm.initial_state([encoded[-1]])
	s = dec_lstm.initial_state()
	c_t_minus_1 = dy.vecInput(state_size*2)

	out = []
	count_EOS = 0
	for i in range(len(in_seq)*2):
		if count_EOS == 1: break
		# w1dt can be computed and cached once for the entire decoding phase
		w1dt = w1dt or w1 * input_mat
		vector = dy.concatenate([last_output_embeddings, c_t_minus_1])
		s = s.add_input(vector)
		h_t = s.output()
		c_t = attend(input_mat, s, w1dt)
		out_vector = dy.affine_transform([b, w, dy.concatenate([h_t, c_t])])
		probs = dy.softmax(out_vector).vec_value()
		next_word = probs.index(max(probs))
		last_output_embeddings = output_lookup[next_word]
		c_t_minus_1 = c_t
		if next_word == EOS:
			count_EOS += 1
			continue

		out.append(english_word_vocab.i2w[next_word])
	return " ".join(out[1:])

#Get the loss for one sentence-pair
def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    #print "Encoded: ", encoded
    return decode(dec_lstm, encoded, output_sentence)

#Train the model
def train(model, training_sentences, num_epochs):
    trainer = dy.SimpleSGDTrainer(model)
    for i in xrange(num_epochs):
    	#Random shuffle of the training sample
    	random.shuffle(training_sentences)
    	num_samples = 0

    	for (input_sentence, output_sentence) in training_sentences:
			num_samples += 1
			loss = get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
			loss_value = loss.value()
			loss.backward()
			trainer.update()
			if num_samples % 1000 == 0:
				print "Epoch Number: ", i
				print "Number of samples: ", num_samples
				print(loss_value)
				trans = generate(input_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
				print trans
				to_write = "{ Epoch Number : " + str(i) + " Sample : " + str(num_samples) + " loss_value : " + str(loss_value) + " output : " + str(trans) + " }"
				f = open("translations_big_no_batch.txt", "a")
				f.write(to_write + "\n")
				f.close()
			if num_samples % 96000 == 0:
				model_filename = "model_big_no_batch/model_epoch_" + str(i) + "_sample_" + str(num_samples)
				model.save(model_filename, [enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup, attention_w1, attention_w2, attention_v, decoder_w, decoder_b])

train(model, indexed_train, 20)









