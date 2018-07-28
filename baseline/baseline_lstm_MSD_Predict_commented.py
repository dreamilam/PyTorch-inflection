from sys import argv, stdout
from random import random, seed, shuffle
from functools import wraps

sysargv = [a for a in argv]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from data import read_dataset, UNK, EOS, NONE, WF, LEMMA, MSD

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
WFDROPOUT=0.1
LSTMDROPOUT=0.3
# Every epoch, we train on a subset of examples from the train set,
# namely, 30% of them randomly sampled.
SAMPLETRAIN=0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to initialize embeddings for wf, lemma, char and msd
def init_model(wf2id,lemma2id,char2id,msd2id):
    global character_lookup,word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, output_lookup

    character_lookup = nn.Embedding(len(char2id), EMBEDDINGS_SIZE)
    word_lookup = nn.Embedding(len(wf2id), EMBEDDINGS_SIZE)
    lemma_lookup = nn.Embedding(len(lemma2id), EMBEDDINGS_SIZE)
    msd_lookup = nn.Embedding(len(msd2id), EMBEDDINGS_SIZE)
    output_lookup = nn.Embedding(len(char2id), EMBEDDINGS_SIZE)

# A simple encoder RNN that takes an input of size input_size and outputs the hidden state
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, LSTM_NUM_OF_LAYERS, bidirectional=bidirectional)
        self.hidden = self.initHidden()

    def forward(self, encoder_input, hidden):
        output = encoder_input.view(1, 1, -1)
        output, self.hidden = self.lstm(output, hidden)
        return output, self.hidden

    def initHidden(self):
        return (torch.zeros(2*LSTM_NUM_OF_LAYERS, 1, self.hidden_size, device=device), 
                torch.zeros(2*LSTM_NUM_OF_LAYERS, 1, self.hidden_size, device=device))

# Decoder to take the encoder inputs and classify it into one of the possible MSDs
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size * 6, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs, hidden):
        encoder_outputs = encoder_outputs.unsqueeze(0)

        output, hidden = self.lstm(encoder_outputs, hidden)
        output = F.softmax(self.out(output), dim=2)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))

def iscandidatemsd(msd):
    """ We only consider nouns, verbs and adjectives. """
    return msd.split(';')[0] in ['N','V','ADJ']

# encoding the lemma in question and return the last hidden state
def encode(embedded, encoder):
    encoder.zero_grad()
    encoder.hidden = encoder.initHidden()
    encoder_output = torch.zeros(1, 2*encoder.hidden_size)

    for ei in range(len(embedded)):
        encoder_output, encoder.hidden = encoder(embedded[ei], encoder.hidden)
        
    return encoder_output

# Function to decode the encoder outputs and calculate the loss 
def decode(encoder_outputs, output_msd, decoder):

    loss = torch.zeros(1)
    decoder.zero_grad()
    msd_index = msd2id[output_msd]
 
    decoder_hidden = decoder.initHidden()
    decoder_output, _ = decoder(encoder_outputs, decoder_hidden)

    loss = -torch.log(decoder_output[0][0][msd_index]).view(1)

    return loss

# Function to concatenate the context vector to each of the character embeddings for the lemma in question
def embed(lemma,context):
    """ Get word embedding and character based embedding for the input
        lemma. Concatenate the embeddings with a context representation. """
    lemma = [EOS] + list(lemma) + [EOS]
    lemma = [c if c in char2id else UNK for c in lemma]
    lemma = [char2id[c] for c in lemma]

    global character_lookup

    return [torch.cat([character_lookup(torch.tensor([c])), context], 1) for c in lemma]

def dropitem(item,item2id,training):
    return item2id[UNK if not item in item2id 
                   or training and random() < WFDROPOUT else
                   item]

# Function which returns the context for a particular lemma
def get_context(i,s,training=0):
    """ Embed context lemma."""
    lemma = s[i][LEMMA] 
    lemma = torch.LongTensor([dropitem(lemma,lemma2id,training)])
    return lemma_lookup(lemma)

# Returns the context i,e the word, lemma and MSD embeddings for a word in a sentence.
def get_sentence_context(i,s, training=1):
    word = s[i][WF] if i > 0 else EOS
    lemma = s[i][LEMMA] if i > 0 else EOS
    msd = s[i][MSD] if i > 0 else EOS

    word = torch.LongTensor([dropitem(word, wf2id, training)])
    lemma = torch.LongTensor([dropitem(lemma, lemma2id, training)])
    msd = torch.LongTensor([dropitem(msd, msd2id, training)])

    return torch.cat((word_lookup(word),lemma_lookup(lemma), msd_lookup(msd)), 1)

# Loops through all the words to the left of the current word,
# gets the context( lemma, wf and MSD embeddings) and sequentially feeeds it to an LSTM
# Returns only the final state of the LSTM
def get_left_encoding(i, s, encoder_left):
    encoder_left.zero_grad()
    encoder_left_hidden = encoder_left.initHidden()
    encoder_left_output = torch.zeros(1, 1, 2*encoder_left.hidden_size)
    for j in range(0, i):
        embedded= get_sentence_context(j,s)
        encoder_left_output, encoder_left_hidden = encoder_left(embedded, encoder_left_hidden)
    return encoder_left_output

# Loops through all the words to the right of the current word,
# gets the context( lemma, wf and MSD embeddings) and sequentially feeeds it to an LSTM
# Returns only the final state of the LSTM
def get_right_encoding(i, s, encoder_right):
    encoder_right.zero_grad()
    encoder_right_hidden = encoder_right.initHidden()
    encoder_right_output = torch.zeros(1, 1, 2*encoder_right.hidden_size)
    for j in range(i+1, len(s)):
        embedded= get_sentence_context(j,s)
        encoder_right_output, encoder_right_hidden = encoder_right(embedded, encoder_right_hidden)
    return encoder_right_output

# Function to predict the MSD of the current word and get the loss       
def get_loss(i, s, encoder, decoder, encoder_left, encoder_right):

    # Get the context vector, Here it is just the lemma embedding of the current lemma
    context = get_context(i,s,training=1)

    # Concatenate the context vector to the character embeddings of the current lemma
    embedded = embed(s[i][LEMMA], context)

    # Feed the embeddings through an lstm and get the LSTM encodings
    encoder_output = encode(embedded, encoder).squeeze(0)

    # Get the LSTM output for the left side
    encoded_left = get_left_encoding(i,s, encoder_left).squeeze(0)

    # Get the LSTM output for the right side
    encoded_right = get_right_encoding(i,s, encoder_right).squeeze(0)

    # Concatenate left, right and the current word encodings
    encoder_outputs = torch.cat((encoder_output, encoded_left, encoded_right), 1)

    # Decode and get the loss
    loss =  decode(encoder_outputs, s[i][MSD], decoder)
    return loss

# Function to evaluate on the dev set
def eval(devdata,id2char,encoder, decoder, encoder_left, encoder_right, generating=1, outf=None):
    _input, gold = devdata
    for n,s in enumerate(_input):
        for i,fields in enumerate(s):
            wf, lemma, msd = fields            
            if msd == NONE and lemma != NONE:
                if generating:
                    msd_pred = generate(i,s,id2char, encoder, decoder, encoder_left, encoder_right)
                    outf.write('%s\n' % '\t'.join([lemma,gold[n][i][WF],msd_pred]))


#  Generating MSDs for new inputs, once the model is trained
def generate(i, s, id2char, encoder, decoder, encoder_left, encoder_right):
    """ Generate a word form for the lemma at position i in sentence s. """
    with torch.no_grad():
        context = get_context(i,s)
        embedded = embed(s[i][LEMMA],context)
        encoder_outputs = encode(embedded, encoder).squeeze(0)
        encoded_left = get_left_encoding(i,s, encoder_left).squeeze(0).repeat(encoder_outputs.size(0), 1)
        encoded_right = get_right_encoding(i,s, encoder_right).squeeze(0).repeat(encoder_outputs.size(0), 1)

        encoder_outputs = torch.cat((encoder_outputs, encoded_left, encoded_right), 1)

        decoder_input = encoder_outputs
        decoder_hidden = decoder.initHidden()

        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        probs = decoder_output.view(-1)
        max_, idx = probs.max(0)
        msd_pred = id2msd[idx.item()]
        
    return msd_pred

def train(traindata,devdata,wf2id,lemma2id,char2id,id2char,msd2id,epochs=20):

    # LSTM to encode the current lemma
    encoder = EncoderRNN(2*EMBEDDINGS_SIZE, STATE_SIZE, bidirectional=True)

    # LSTM to encode the part of the sentence to the left of the current lemma
    encoder_left = EncoderRNN(3*EMBEDDINGS_SIZE, STATE_SIZE, bidirectional=True)

    # LSTM to encode the part of the sentence to the right of the current lemma
    encoder_right = EncoderRNN(3*EMBEDDINGS_SIZE, STATE_SIZE, bidirectional=True)

    # LSTM that takes the concatenated encoder inputs and classifies into one of the MSDs
    decoder = DecoderRNN(STATE_SIZE, len(id2msd), 0.3)
    
    # Optimizers for the encoders and decoders
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0005)
    encoder_left_optimizer = optim.Adam(encoder_left.parameters(), lr=0.0005)
    encoder_right_optimizer = optim.Adam(encoder_right.parameters(), lr=0.0005)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0005)
    
    # Training
    for epoch in range(epochs):
        print("EPOCH %u" % (epoch + 1))
        shuffle(traindata)
        total_loss = 0
        for n,s in enumerate(traindata):
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                stdout.write("Example %u of %u\r" % (n+1,len(traindata)))
                
                # Check if the word is a verb, noun or an adjective
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE)):
                    # Predict the MSD of the current word and get the loss
                    loss = get_loss(i,s, encoder, decoder, encoder_left, encoder_right)
                    loss_value = loss.item()

                    # Back Propagate
                    loss.backward()

                    # Step through the optimizer
                    encoder_optimizer.step()
                    encoder_left_optimizer.step()
                    encoder_right_optimizer.step()
                    decoder_optimizer.step()
                    total_loss += loss_value
        print("\nLoss per sentence: %.3f" % (total_loss/len(traindata)))
        print("Example outputs:")

        # Print some MSD predictions to the console
        for s in traindata[:10]:
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE)):
                #    and random() < SAMPLETRAIN:
                    print("Gold_msd:", s[i][MSD], "Output:",
                          generate(i,s,id2char, encoder, decoder, encoder_left, encoder_right),
                          "WF:",wf)
                    break

        # File to write the Predictions to
        outf = open('sv-high-dev', 'w')

        # Evaluate on dev data
        eval(devdata,id2char,encoder, decoder, encoder_left, encoder_right, 1, outf)
        outf.close()

if __name__=='__main__':

    # Read the training data and store each sentence as a list
    # Store the word forms, lemmas, chars, msds in lookup dictionaries
    traindata, wf2id, lemma2id, char2id, msd2id = read_dataset(sysargv[1])

    # Read the dev set data
    devinputdata, _, _, _, _ = read_dataset(sysargv[2])

    # Read the Gold data
    devgolddata, _, _, _, _ = read_dataset(sysargv[3])

    # Lookup dictionary from index to char
    id2char = {id:char for char,id in char2id.items()}

    # Lookup dictionary from index to msd
    id2msd = {id:msd for msd,id in msd2id.items()}

    # Initialize embeddings for word forms, lemmas, chars and msds
    init_model(wf2id,lemma2id,char2id,msd2id)

    # Train and evaluate
    train(traindata,[devinputdata,devgolddata], wf2id,lemma2id,char2id,id2char,msd2id,10)    
