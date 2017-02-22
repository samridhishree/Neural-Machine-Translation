from collections import Counter, defaultdict
from itertools import count
import random
import _dynet as dy
import numpy as np
import os
import sys
import nltk
from operator import itemgetter

dyparams = dy.DynetParams()
dyparams.set_mem(4000)
dyparams.init()

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
german_test_file = sys.argv[3]
model_file = sys.argv[4]
lstm_num_of_layers = 2
embeddings_size = 300
state_size = 150
attention_size = 300

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

model = dy.Model()
print "Loading Model."
[enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup, attention_w1, attention_w2, attention_v, decoder_w, decoder_b] = model.load(model_file)
print "Model Loaded."

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


#Generate the translations from the current trained state of the model
#Generate the translations from the current trained state of the model
def generate_greedy(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)   
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = w1 * input_mat

    last_output_embeddings = output_lookup[BOS]
    #s = dec_lstm.initial_state([encoded[-1]])
    s = dec_lstm.initial_state()
    c_t_minus_1 = dy.vecInput(state_size*2)

    out = []
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 1: break
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


#Generate with beam search
def generate_beam(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, beam_size):
    embedded = embed_sentence(in_seq)   
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    #Initialize beam
    final_beam = [{'prefix': [BOS], 'value': 0, 'c_t_minus_1': dy.vecInput(state_size*2)} for i in range(beam_size)]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = w1 * input_mat

    last_output_embeddings = output_lookup[BOS]
    s = dec_lstm.initial_state()
    c_t_minus_1 = dy.vecInput(state_size*2)

    out = []
    count_EOS = 0
    for i in range(len(in_seq)*2):
        temp_beam = []
        for item in final_beam:
            num_eos_item = item['prefix'].count(EOS)
            if num_eos_item == 1:
                temp_beam.append(item)
                continue

            last_word = item['prefix'][-1]
            last_prob = item['value']
            last_output_embeddings = output_lookup[last_word]
            c_t_minus_1 = item['c_t_minus_1']
            
            vector = dy.concatenate([last_output_embeddings, c_t_minus_1])
            s = s.add_input(vector)
            h_t = s.output()
            c_t = attend(input_mat, s, w1dt)

            out_vector = dy.affine_transform([b, w, dy.concatenate([h_t, c_t])])
            probs = dy.log_softmax(out_vector).vec_value()
            temp_probs = [{'index': i, 'value': prob} for i,prob in enumerate(probs)]
            temp_probs = sorted(temp_probs, key=itemgetter('value'), reverse=True)
            top_k_probs = temp_probs[:beam_size]
            #print "top-k probs = ", top_k_probs
            for prob_item in top_k_probs:
                temp_beam.append({'prefix': item['prefix'] + [prob_item['index']], 'value': last_prob + prob_item['value'], 'c_t_minus_1': c_t})

        temp_beam = sorted(temp_beam, key=itemgetter('value'), reverse=True)
        final_beam = temp_beam[:beam_size]

        num_eos_item = [val['prefix'].count(EOS) for val in final_beam]
        if sum(num_eos_item) == beam_size:
            break

    max_prob_sequence = final_beam[0]['prefix']
    for word in max_prob_sequence[1:-1]:
        out.append(english_word_vocab.i2w[word])
    return " ".join(out)

def ReadValidationSet(ger_file_valid):
    german_valid = list(read(ger_file_valid))
    print "german_valid length = ", len(german_valid)
    indexed_source = []
    german_lists = [ger.strip().split(" ") for ger in german_valid]
    
    #Create the indexed version of the list
    for german_list in german_lists:
        idx_german_list =[]
        for word in german_list:
            if word in german_word_vocab.w2i.keys():
                idx_german_list.append(german_word_vocab.w2i[word])
            else:
                idx_german_list.append(german_word_vocab.w2i["<UNK>"])

        indexed_source.append(idx_german_list)

    print "source_sentences length = ", len(indexed_source)
    return indexed_source

#Generate the translations
source_sentences = ReadValidationSet(german_test_file)

#Greedy search
f = open('hypotheses_greedy.txt', "a")
for sentence in source_sentences:
    hyp = generate_greedy(sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
    f.write(hyp + "\n")
f.close()

#Beam Search
f1 = open("hypotheses_beam.txt", "a")
for sentence in source_sentences:
    hyp = generate_beam(sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
    f.write(hyp + "\n")
f1.close()







