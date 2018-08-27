from sys import argv, stdout
from random import random, seed, shuffle
from functools import wraps

sysargv = [a for a in argv]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

from data import read_dataset, UNK, EOS, NONE, WF, LEMMA, MSD

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
ATTENTION_SIZE = 100
WFDROPOUT=0.1
LSTMDROPOUT=0.3
# Every epoch, we train on a subset of examples from the train set,
# namely, 30% of them randomly sampled.
SAMPLETRAIN=0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(wf2id,lemma2id,char2id,msd2id):
    global character_lookup,word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, output_lookup

    character_lookup = nn.Embedding(len(char2id), EMBEDDINGS_SIZE)
    word_lookup = nn.Embedding(len(wf2id), EMBEDDINGS_SIZE)
    lemma_lookup = nn.Embedding(len(lemma2id), EMBEDDINGS_SIZE)
    msd_lookup = nn.Embedding(len(msd2id), EMBEDDINGS_SIZE)
    output_lookup = nn.Embedding(len(char2id), EMBEDDINGS_SIZE)

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

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
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

def encode(embedded, encoder_fwd):
    encoder_fwd.zero_grad()
    encoder_fwd.hidden = encoder_fwd.initHidden()
    encoder_fwd_output = torch.zeros(1, 2*encoder_fwd.hidden_size)
    # embedded = Variable(embedded)

    for ei in range(len(embedded)):
        encoder_fwd_output, encoder_fwd.hidden = encoder_fwd(embedded[ei], encoder_fwd.hidden)
        
    return encoder_fwd_output

def decode(encoder_outputs, output_msd, decoder):

    loss = torch.zeros(1)
    decoder.zero_grad()
    msd_index = msd2id[output_msd]
 
    decoder_hidden = decoder.initHidden()
    decoder_output, _ = decoder(encoder_outputs, decoder_hidden)

    loss = -torch.log(decoder_output[0][0][msd_index]).view(1)

    return loss

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

def embed_context(prevword,prevlemma,prevmsd,lemma,
                  nextword,nextlemma,nextmsd):
    """ Emebed context elements. """
    
    return torch.cat((word_lookup(prevword), word_lookup(nextword),
                           lemma_lookup(prevlemma), lemma_lookup(nextlemma),
                           msd_lookup(prevmsd),msd_lookup(nextmsd),
                           lemma_lookup(lemma)), 1)

def get_context(i,s,training=0):
    """ Embed context words, lemmas and MSDs. 
        The context of a lemma consists of the previous and following
        word forms, lemmas and MSDs as well as the MSD for the lemma
        in question.
    """
    lemma = s[i][LEMMA] 
    lemma = torch.LongTensor([dropitem(lemma,lemma2id,training)])

    return lemma_lookup(lemma)

def get_sentence_context(i,s, training=1):
    word = s[i][WF] if i > 0 else EOS
    lemma = s[i][LEMMA] if i > 0 else EOS
    msd = s[i][MSD] if i > 0 else EOS

    word = torch.LongTensor([dropitem(word, wf2id, training)])
    lemma = torch.LongTensor([dropitem(lemma, lemma2id, training)])
    msd = torch.LongTensor([dropitem(msd, msd2id, training)])

    return torch.cat((word_lookup(word),lemma_lookup(lemma), msd_lookup(msd)), 1)


def get_left_encoding(i, s, encoder_left):
    encoder_left.zero_grad()
    encoder_left_hidden = encoder_left.initHidden()
    encoder_left_output = torch.zeros(1, 1, 2*encoder_left.hidden_size)
    for j in range(0, i):
        embedded= get_sentence_context(j,s)
        encoder_left_output, encoder_left_hidden = encoder_left(embedded, encoder_left_hidden)
    return encoder_left_output

def get_right_encoding(i, s, encoder_right):
    encoder_right.zero_grad()
    encoder_right_hidden = encoder_right.initHidden()
    encoder_right_output = torch.zeros(1, 1, 2*encoder_right.hidden_size)
    for j in range(i+1, len(s)):
        embedded= get_sentence_context(j,s)
        encoder_right_output, encoder_right_hidden = encoder_right(embedded, encoder_right_hidden)
    return encoder_right_output
        
def get_loss(i, s, encoder_fwd, decoder, encoder_left, encoder_right):

    context = get_context(i,s,training=1)
    embedded = embed(s[i][LEMMA], context)
    encoder_output = encode(embedded, encoder_fwd).squeeze(0)

    encoded_left = get_left_encoding(i,s, encoder_left).squeeze(0).repeat(encoder_output.size(0), 1)
    encoded_right = get_right_encoding(i,s, encoder_right).squeeze(0).repeat(encoder_output.size(0), 1)
    encoder_outputs = torch.cat((encoder_output, encoded_left, encoded_right), 1)
    loss =  decode(encoder_outputs, s[i][MSD], decoder)
    return loss

def memolrec(func):
    """Memoizer for Levenshtein."""
    cache = {}
    @wraps(func)
    def wrap(sp, tp, sr, tr, cost):
        if (sr,tr) not in cache:
            res = func(sp, tp, sr, tr, cost)
            cache[(sr,tr)] = (res[0][len(sp):], res[1][len(tp):], res[4] - cost)
        return sp + cache[(sr,tr)][0], tp + cache[(sr,tr)][1], '', '', cost + cache[(sr,tr)][2]
    return wrap

def levenshtein(s, t, inscost = 1.0, delcost = 1.0, substcost = 1.0):
    """Recursive implementation of Levenshtein, with alignments returned.
       Courtesy of Mans Hulden. """
    @memolrec
    def lrec(spast, tpast, srem, trem, cost):
        if len(srem) == 0:
            return spast + len(trem) * '_', tpast + trem, '', '', cost + len(trem)
        if len(trem) == 0:
            return spast + srem, tpast + len(srem) * '_', '', '', cost + len(srem)
        
        addcost = 0
        if srem[0] != trem[0]:
            addcost = substcost
            
        return min((lrec(spast + srem[0], tpast + trem[0], srem[1:], trem[1:], cost + addcost),
                    lrec(spast + '_', tpast + trem[0], srem, trem[1:], cost + inscost),
                    lrec(spast + srem[0], tpast + '_', srem[1:], trem, cost + delcost)),
                   key = lambda x: x[4])
        
    answer = lrec('', '', s, t, 0)
    return answer[0],answer[1],answer[4]

def eval(devdata,id2char,encoder_fwd, decoder, encoder_left, encoder_right, generating=1, outf=None):
    _input, gold = devdata
    corr = 0.0
    lev=0.0
    tot = 0.0
    for n,s in enumerate(_input):
        for i,fields in enumerate(s):
            wf, lemma, msd = fields            
            if msd == NONE and lemma != NONE:
                if generating:
                    msd_pred = generate(i,s,id2char, encoder_fwd, decoder, encoder_left, encoder_right)
                    outf.write('%s\n' % '\t'.join([lemma,gold[n][i][WF],msd_pred]))
                # if wf == gold[n][i][WF]:
                #     corr += 1
        #         lev += levenshtein(wf,gold[n][i][WF])[2]
        #         tot += 1
            # if outf:
            #     outf.write('%s\n' % '\t'.join([lemma,gold[n][i][WF],msd_pred]))
        # if outf:
        #     outf.write('\n')
    return (0,0) if tot == 0 else (corr / tot, lev/tot)

def generate(i, s, id2char, encoder_fwd, decoder, encoder_left, encoder_right):
    """ Generate a word form for the lemma at position i in sentence s. """
    with torch.no_grad():
        context = get_context(i,s)
        embedded = embed(s[i][LEMMA],context)
        encoder_outputs = encode(embedded, encoder_fwd).squeeze(0)
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
    encoder_fwd = EncoderRNN(2*EMBEDDINGS_SIZE, STATE_SIZE, bidirectional=True)
    decoder = AttnDecoderRNN(STATE_SIZE, len(id2msd), 0.3)
    encoder_left = EncoderRNN(3*EMBEDDINGS_SIZE, STATE_SIZE, bidirectional=True)
    encoder_right = EncoderRNN(3*EMBEDDINGS_SIZE, STATE_SIZE, bidirectional=True)

    encoder_fwd_optimizer = optim.Adam(encoder_fwd.parameters(), lr=0.0005)
    encoder_left_optimizer = optim.Adam(encoder_left.parameters(), lr=0.0005)
    encoder_right_optimizer = optim.Adam(encoder_right.parameters(), lr=0.0005)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0005)
    
    for epoch in range(epochs):
        print("EPOCH %u" % (epoch + 1))
        train_idx = int(0.8 * len(traindata))
        actual_traindata = traindata[:train_idx]
        shuffle(actual_traindata)
        total_loss = 0

        for n,s in enumerate(actual_traindata):
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                stdout.write("Example %u of %u\r" % 
                             (n+1,len(traindata)))
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE)):
                #    and random() < SAMPLETRAIN:
                # if (True):
                #    and random() < SAMPLETRAIN:
                    loss = get_loss(i,s, encoder_fwd, decoder, encoder_left, encoder_right)
                    loss_value = loss.item()
                    loss.backward()
                    encoder_fwd_optimizer.step()
                    encoder_left_optimizer.step()
                    encoder_right_optimizer.step()
                    decoder_optimizer.step()
                    total_loss += loss_value
        print("\nLoss per sentence: %.3f" % (total_loss/len(traindata)))
        print("Example outputs:")
        for s in actual_traindata[:10]:
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE)):
                #    and random() < SAMPLETRAIN:
                    print("Gold_msd:", s[i][MSD], "Output:",
                          generate(i,s,id2char, encoder_fwd, decoder, encoder_left, encoder_right),
                          "WF:",wf)
                    break
        corr = 0
        count = 0
        for s in traindata[train_idx:]:
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE)):
                    msd_pred = generate(i,s,id2char, encoder_fwd, decoder, encoder_left, encoder_right)
                    if msd_pred == s[i][MSD]:
                        corr += 1
                    count += 1
        print()
        print(corr, count)
        print("Devset efficiency = ", corr/count*100)
        outf = None
        if epoch == 9:
            outf = open('low-dev', 'w')
            devacc, devlev = eval(devdata,id2char,encoder_fwd, decoder, encoder_left, encoder_right, 1, outf)
            outf.close()
        # print("Development set accuracy: %.2f" % (100*devacc))
        # print("Development set avg. Levenshtein distance: %.2f" % devlev)
        # print()


if __name__=='__main__':
    traindata, wf2id, lemma2id, char2id, msd2id = read_dataset(sysargv[1])
    devinputdata, _, _, _, _ = read_dataset(sysargv[2])
    devgolddata, _, _, _, _ = read_dataset(sysargv[3])

    id2char = {id:char for char,id in char2id.items()}
    id2msd = {id:msd for msd,id in msd2id.items()}
    init_model(wf2id,lemma2id,char2id,msd2id)
    train(traindata,[devinputdata,devgolddata],
          wf2id,lemma2id,char2id,id2char,msd2id,10)    
    # eval([devinputdata,devgolddata],id2char,generating=1,
    #      outf=open("%s-out" % sysargv[2],"w"))
