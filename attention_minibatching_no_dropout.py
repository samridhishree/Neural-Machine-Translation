from collections import Counter, defaultdict
from itertools import count
import random
import _gdynet as dy
import numpy as np
import os
import sys
import nltk
import math
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
dropout = 0.0

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
#german_validation_file = sys.argv[3]
#english_validation_file = sys.argv[4]

#Get the training sentences as a list
german_train = list(read(german_train_file))
english_train = list(read(english_train_file))
#german_valid = list(read(german_validation_file))
#english_valid = list(read(english_validation_file))
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

training_sentences = [[ger.strip().split(" "), eng.strip().split(" ")] for (ger, eng) in zip(german_train, english_train)]
#validation_sentences = [[ger.strip().split(" "), eng.strip().split(" ")] for (ger, eng) in zip(german_valid, english_valid)]

indexed_train = []
for (german_list, english_list) in training_sentences:
	german_list = [german_word_vocab.w2i[word] for word in german_list]
	english_list = [english_word_vocab.w2i[word] for word in english_list]
	indexed_train.append((german_list, english_list))

#Sort the indexed train by the length of german sentences (descending)
indexed_train.sort(key=lambda item:(-len(item[0]), item))

#indexed_valid = []
#validation_num_op_tokens = 0
#for (german_list, english_list) in validation_sentences:
#	indexed_german_list = []
#	indexed_eng_list = []
#	validation_num_op_tokens += len(english_list)
#	for word in german_list:
#		if word in german_word_vocab.w2i.keys():
#			indexed_german_list.append(german_word_vocab.w2i[word])
#		else:
#			indexed_german_list.append(german_word_vocab.w2i["<UNK>"])
#	for word in english_list:
#		if word in english_word_vocab.w2i.keys():
#			indexed_eng_list.append(english_word_vocab.w2i[word])
#		else:
#			indexed_eng_list.append(english_word_vocab.w2i["<UNK>"])
#	indexed_valid.append((indexed_german_list, indexed_eng_list))

#Declare and define the enc-doc models
model = dy.Model()
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

#Convert the input(german) sentence into its embedded form
def embed_sentence(sentence):
	#print "In embed sentence"
	global input_lookup
	return([input_lookup[w] for w in sentence])

#Convert the list of sentences(each word is rep as word-id) to the proper embedded representation
def sentences_to_batch(sentences):
	#print "in sentence to batch"
	wids = []
	masks = []
	max_len = max(map(lambda x: len(x), sentences))
	for i in xrange(max_len):
	    wids.append([(sent[i] if len(sent)>i else EOS) for sent in sentences])
	    mask = [(1 if len(sent)>i else 0) for sent in sentences]
	    masks.append(mask)
	return wids, masks

#Run the bi-directional encoding for the sentences in the batch
def encode_batch(enc_fwd_lstm, enc_bwd_lstm, sentences):
	#print "in encode batch"
	global input_lookup
	
	input_words, masks = sentences_to_batch(sentences)
	input_embeddings = [dy.lookup_batch(input_lookup, wids) for wids in input_words]
	input_embeddings_rev = input_embeddings[::-1]

    fwd_state = enc_fwd_lstm.initial_state()
	bwd_state = enc_bwd_lstm.initial_state()

	#Get the forward and backward encodings
	fwd_vectors = fwd_state.transduce(input_embeddings)
	bwd_vectors = bwd_state.transduce(input_embeddings_rev)
    bwd_vectors = bwd_vectors[::-1]

	input_vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
	return input_vectors

#Return the context after calculating attention : MLP method
def attend_batch(input_mat, state, w1dt, batch_size, input_length):
	#print "in attend batch"
	global attention_w2
	global attention_v
	w2 = dy.parameter(attention_w2)
	v = dy.parameter(attention_v)
        #print "Calculating w2dt"
	w2dt = w2*dy.concatenate(list(state.s()))
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        attention_reshaped = dy.reshape(unnormalized, (input_length, ), batch_size)
	att_weights = dy.softmax(unnormalized)
	context = input_mat * att_weights
	return context

#Decoder batch - perform decoding for batch
def decode_batch(dec_lstm, input_encodings, output_sentences):
	#print "in decode batch"
	w = dy.parameter(decoder_w)
	b = dy.parameter(decoder_b)
	w1 = dy.parameter(attention_w1)

	output_words, masks = sentences_to_batch(output_sentences)
	#decoder_target_input = zip(output_words[1:], masks[1:])

	batch_size = len(output_sentences)
	input_length = len(input_encodings)

	input_mat = dy.concatenate_cols(input_encodings)
        #print "Computing w1dt"
	w1dt = w1 * input_mat

	s = dec_lstm.initial_state()
	c_t_minus_1 = dy.vecInput(state_size * 2)
	loss = []

	for t in range(1, len(output_words)):
	    last_output_embeddings = dy.lookup_batch(output_lookup, output_words[t-1])
	    vector = dy.concatenate([c_t_minus_1, last_output_embeddings])
	    s = s.add_input(vector)
	    h_t = s.output()
            #print "Calling attend"
	    c_t = attend_batch(input_mat, s, w1dt, batch_size, input_length)
	    predicted = w * dy.concatenate([h_t, c_t]) + b
	    if(dropout > 0.):
		predicted = dy.dropout(predicted, dropout)
	    cur_loss = dy.pickneglogsoftmax_batch(predicted, output_words[t])
        c_t_minus_1 = c_t

	    #Mask the loss in case mask == 0
	    if 0 in masks[t]:
		mask = dy.inputVector(masks[t])
		mask = dy.reshape(mask, (1, ), batch_size)
		cur_loss = cur_loss * mask

	    loss.append(cur_loss)
	
	#Get the average batch loss
	loss = dy.esum(loss)
	loss = dy.sum_batches(loss) / batch_size
	return loss

#Generate the translations from the current trained state of the model
def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
	#print "in generate"
	dy.renew_cg()
	embedded = embed_sentence(in_seq)
	encoded = encode_batch(enc_fwd_lstm, enc_bwd_lstm, embedded)

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
		c_t = attend_batch(input_mat, s, w1dt, 1, 1)
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

#Get the loss for the batch
def get_batch_loss(input_sentences, output_sentences, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
	#print "in batch loss"
	dy.renew_cg()
	encoded_batch = encode_batch(enc_fwd_lstm, enc_bwd_lstm, input_sentences)
	return decode_batch(dec_lstm, encoded_batch, output_sentences)


def generate_batches(training_sentences, batch_size):
	minibatched_train_sents = [training_sentences[i:i+batch_size] for i in range(0, len(training_sentences), batch_size)]
        minibatched_train_sents.append(training_sentences[i:])
	rearranged_batches = []
	for batch in minibatched_train_sents:
		source = []
 		target = []
 		for sample in batch:
 			source.append(sample[0])
 			target.append(sample[1])
		rearranged_batches.append([source, target])
	return rearranged_batches

#def validation_perplexity(validation_batches):
    #cum_loss = 0
#	for batch in validation_batches:
#		input_sentences = batch[0]
#		output_sentences = batch[1]
#		batch_size = len(output_sentences)
#		loss = get_batch_loss(input_sentences, output_sentences, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
#		loss_value = loss.value()
#		cum_loss += (loss_value * batch_size)
#	perplexity = math.exp((cum_loss)/float(validation_num_op_tokens))
#	return perplexity


#Train the model
def train(model, training_batches, num_epochs):
	#print "in train"
	trainer = dy.SimpleSGDTrainer(model)
	data_size = len(indexed_train)
	for i in xrange(num_epochs):
	    random.shuffle(training_batches)
	    num_batches = 0
	    cum_loss = 0
	    num_samples = 0

	    for batch in training_batches:
		input_sentences = batch[0]
		output_sentences = batch[1]
		batch_size = len(output_sentences)
                num_batches += 1
		loss = get_batch_loss(input_sentences, output_sentences, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
		loss_value = loss.value()
                loss.backward()
		trainer.update()

		cum_loss = cum_loss + (loss_value * batch_size)
		num_samples += batch_size
		#perplexity = math.exp((loss_value * batch_size)/float(sum(len(s) for s in output_sentences)))
		#validation_ppl = validation_perplexity(validation_batches)
		if num_batches % 100 == 0:
                    #validation_ppl = validation_perplexity(validation_batches)
                    perplexity = math.exp(float(loss_value * batch_size)/float(sum(len(s) for s in output_sentences)))
		    print "Epoch Number: ", i
                    print "Current Batch Loss: %f, Cumulative loss: %f, Train Perplexity: %f" % (loss_value, cum_loss, perplexity)
                    to_write = "{ Epoch Number : " + str(i) + " Sample : " + str(num_samples) + " Curent Batch Perplexity : " + str(perplexity) + " }"
		    f = open("translations_batch_nodrop_512.txt", "a")
		    f.write(to_write + "\n")
		    f.close()
                if num_batches % 1000 == 0:
		    model_filename = "model_batch_nodrop_512/model_epoch_" + str(i) + "_sample_" + str(num_samples)
		    model.save(model_filename, [enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup, attention_w1, attention_w2, attention_v, decoder_w, decoder_b])


print "Creating training batch"
train_batches = generate_batches(indexed_train, 32)
#print "Creating generate batch"
#validation_batches = generate_batches(indexed_valid, 5)
print "Starting training"
train(model, train_batches, 30)

