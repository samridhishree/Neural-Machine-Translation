from collections import Counter, defaultdict
from itertools import count
import random
import _dynet as dy
import numpy as np
import os
import sys

dyparams = dy.DynetParams()
dyparams.set_mem(2500)
dyparams.init()

#Vocabulary related parameters
german_vocab_size = 0
english_vocab_size = 0
EOS = 0
#Hyper-parameters of the model
lstm_num_of_layers = 2
embeddings_size = 32
state_size = 32
attention_size = 32

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
def read(fname):
    fh = file(fname)
    for line in fh:
        sent = "<s> " + line.strip() + " <s>"
        yield sent

german_train_file = sys.argv[1]
english_train_file = sys.argv[2]

#Get the training sentences as a list
german_train = list(read(german_train_file))
english_train = list(read(english_train_file))
german_word_corpus = []
english_word_corpus = []

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
EOS = english_word_vocab.w2i["<s>"]

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
enc_fwd_lstm = dy.LSTMBuilder(lstm_num_of_layers, embeddings_size, state_size, model)
enc_bwd_lstm = dy.LSTMBuilder(lstm_num_of_layers, embeddings_size, state_size, model)
dec_lstm = dy.LSTMBuilder(lstm_num_of_layers, ((state_size * 2) + embeddings_size), state_size, model)

#Define the model parameters
input_lookup = model.add_lookup_parameters((german_vocab_size, embeddings_size))
attention_w1 = model.add_parameters(((attention_size, (state_size * 2))))
attention_w2 = model.add_parameters(((attention_size, (state_size * lstm_num_of_layers * 2))))
attention_v = model.add_parameters((1, attention_size))
decoder_w = model.add_parameters((english_vocab_size, state_size))
decoder_b = model.add_parameters((english_vocab_size))
output_lookup = model.add_lookup_parameters((english_vocab_size, embeddings_size))

#Convert the input(german) sentence into its embedded form
def create_batch(sentences):
    wids = []
    masks = []
    max_len = max([len(x) for x in sentences])
    for i in range(max_len):
        wids.append([(sent[i] if len(sent)>i else EOS) for sent in sentences])
        mask = [(1 if len(sent)>i else 0) for sent in sentences]
        masks.append(mask)
    print "wids: ", wids
    print "masks: ", masks
    return wids, masks

#Run the lstm for the input batch
def run_lstm_batch(init_state, wids):
    output_wids = []
    #init_ids = [EOS] * batch_size
    #init_state = init_state.add_input(dy.lookup_batch(input_lookup,init_ids))
    #output_wids.append(init_state.output())
    for sample in wids:
        print "sample: ", sample
        init_state = init_state.add_input(dy.lookup_batch(input_lookup, sample))
        output_wids.append(init_state.output())
        #output_wids = init_state.output()
    return output_wids


#Runs the bi-directional encoder for each word of the sentence passed.
def encode_batch(enc_fwd_lstm, enc_bwd_lstm, sentences):
    reversed_sentences = []
    for sent in sentences:
        reversed_sentences.append(list(reversed(sent)))
    print "Reversed: ", reversed_sentences

    fwd_batch_wids, fwd_mask = create_batch(sentences)
    bwd_batch_wids, bwd_mask = create_batch(reversed_sentences)

    #Fwd-bwd encodings
    fwd_vectors = run_lstm_batch(enc_fwd_lstm.initial_state(), fwd_batch_wids)
    bwd_vectors = run_lstm_batch(enc_bwd_lstm.initial_state(), bwd_batch_wids)


    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
    print "Vectors: ", vectors
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
def decode(dec_lstm, vectors, output_sentences):
    #Convert the words to word-ids
    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)  #8888888888
    w1dt = None

    output_wids, masks = create_batch(output_sentences)

    init_last = [EOS] * len(output_sentences)
    last_output_embeddings = dy.lookup_batch(output_lookup, init_last)
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(state_size*2), last_output_embeddings]))  #8888
    losses = []

    for wid, mask in zip(output_wids, masks):
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat   #8888888
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])   #h(t,e) = enc([embed(e(t-1);c(t-1))])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        #probs = dy.softmax(out_vector)
        
        #Loss, masking
        loss = dy.pickneglogsoftmax_batch(probs, wid)
        # mask the loss if at least one sentence is shorter
        mask_expr = dy.inputVector(mask)
        mask_expr = dy.reshape(mask_expr, (1,), batch_size)
        loss = loss * mask_expr

        losses.append(loss)
        last_output_embeddings = dy.lookup_batch(output_lookup, wid)
        
    total_loss = dy.sum_batches(dy.esum(losses))
    return total_loss

#Generate the translations from the current trained state of the model
def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    #Generate embeddings for the input sentence
    embedded = embed_sentence(in_seq)   
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[EOS]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(state_size * 2), last_output_embeddings]))

    out = []
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_word = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_word]
        if next_word == EOS:
            count_EOS += 1
            continue

        out.append(english_word_vocab.i2w[next_word])
    return " ".join(out)

#Get the loss for one sentence-pair
def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)

#Train the model
def train(model, training_sentences, num_epochs):
    trainer = dy.SimpleSGDTrainer(model)
    for i in xrange(num_epochs):
        for (input_sentence, output_sentence) in training_sentences:
            loss = get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            if i % 20 == 0:
                print(loss_value)
                print(generate(input_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm))

#train(model, indexed_train, 600)
english_sentences = []
german_sentences = []
for (german,english) in indexed_train:
    german_sentences.append(german)
    english_sentences.append(english)

#print "english sentences: ", english_sentences
#english_sentences.sort(key=lambda x: -len(x))
#create_batch(english_sentences)
print "german_sentences: ", german_sentences
encode_batch(enc_fwd_lstm, enc_bwd_lstm, german_sentences)



