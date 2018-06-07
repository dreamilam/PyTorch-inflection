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
ATTENTION_SIZE = 100
WFDROPOUT=0.1
LSTMDROPOUT=0.3
# Every epoch, we train on a subset of examples from the train set,
# namely, 30% of them randomly sampled.
SAMPLETRAIN=0.3

def init_model(wf2id,lemma2id,char2id,msd2id):
    global character_lookup,word_lookup, lemma_lookup, msd_lookup, attention_w1, attention_w2, output_lookup

    character_lookup = nn.Embedding(len(char2id), EMBEDDINGS_SIZE)
    word_lookup = nn.Embedding(len(wf2id), EMBEDDINGS_SIZE)
    lemma_lookup = nn.Embedding(len(lemma2id), EMBEDDINGS_SIZE)
    msd_lookup = nn.Embedding(len(msd2id), EMBEDDINGS_SIZE)
    output_lookup = nn.Embedding(len(char2id), EMBEDDINGS_SIZE)

class LSTMEncoder(nn.Module):
    def __init__(self, layers, input_size, state_size):
        super(LSTMEncoder, self).__init__()
        self.state_size = state_size
        self.lstm = nn.LSTM(input_size, state_size, layers)
        self.state = self.init_state()
    
    def init_state(self):
        return (torch.zeros(1, 1, self.state_size),
                torch.zeros(1,1, self.state_size))
    
    def forward(self, input_vecs):
        out_vectors = []
        for vector in input_vecs:
            out_vector, self.state = self.lstm(vector.view(1, 1, -1), self.state)
            out_vectors.append(out_vector)
        return out_vectors

class AttnLSTMDecoder(nn.Module):
    def __init__(self, layers, input_size, state_size):
        super(AttnLSTMDecoder, self).__init__()
        self.state_size = state_size
        self.lstm = nn.LSTM(input_size, state_size, layers)
        self.state = self.init_state()
        self.w1 = nn.Linear(STATE_SIZE*2, ATTENTION_SIZE, bias=False)
        self.w2 = nn.Linear(STATE_SIZE*2*LSTM_NUM_OF_LAYERS, ATTENTION_SIZE, bias=False)
        self.v = nn.Linear(ATTENTION_SIZE, 1)
        self.linear = nn.Linear(STATE_SIZE, len(char2id))

    def init_state(self):
        return (torch.zeros(1,1, self.state_size), torch.zeros(1,1, self.state_size))

    def attend(self, input_mat, w1dt):
        w2dt = self.w2(torch.cat(self.state, 2))
        unnormalized = self.v(F.tanh(torch.add(w1dt, w2dt)))

        att_weights = F.softmax(unnormalized.view(-1,1))
        input_matrix = torch.t(input_mat.view(-1,2*ATTENTION_SIZE))
        context = input_matrix.mm(att_weights)

        return context

    def forward(self, vectors, output):
        output = [EOS] + list(output) + [EOS]
        output = [char2id[c] for c in output]
        loss = torch.zeros(1)

        input_mat = torch.cat(vectors)

        w1dt = None 
        last_output_embeddings = output_lookup(torch.tensor([char2id[EOS]]))
        initial_input = torch.cat((torch.zeros(STATE_SIZE*2).view(1,-1), last_output_embeddings), 1)
        _, self.state = self.lstm(initial_input.view(1, 1, -1), self.state)
        
        w1dt = self.w1(torch.t(input_mat))
        for char in output:
            # # w1dt can be computed and cached once for the entire decoding phase
            # if not w1dt:
            #     w1dt = self.w1(torch.t(input_mat))
            vector = torch.cat((self.attend(input_mat, w1dt), last_output_embeddings.view(-1,1)))
            out, self.state = self.lstm(vector.view(1, 1, -1), self.state)
            out_vector = self.linear(out)
            probs = F.softmax(out_vector.view(1,-1)).view(-1)

            last_output_embeddings = output_lookup(torch.tensor([char]))

            loss = torch.cat((loss, -torch.log(probs[char]).view(1)))
        loss = torch.sum(loss)
        return loss


def iscandidatemsd(msd):
    """ We only consider nouns, verbs and adjectives. """
    return msd.split(';')[0] in ['N','V','ADJ']

def encode(embedded, encoder_fwd, encoder_bwd):
    embedded_rev = list(reversed(embedded))

    encoder_fwd.zero_grad()
    encoder_fwd.state = encoder_fwd.init_state()
    fwd_vectors = encoder_fwd(embedded)

    encoder_bwd.zero_grad()
    encoder_bwd.state = encoder_bwd.init_state()
    bwd_vectors = encoder_bwd(embedded_rev)

    vectors = [torch.cat(list(p), 2) for p in zip(fwd_vectors, bwd_vectors)]
    return vectors

def decode(vectors, output, decoder):
    decoder.zero_grad()
    decoder.state = decoder.init_state()
    out = decoder(vectors, output)
    return out

def embed(lemma,context):
    """ Get word embedding and character based embedding for the input
        lemma. Concatenate the embeddings with a context representation. """
    lemma = [EOS] + list(lemma) + [EOS]
    lemma = [c if c in char2id else UNK for c in lemma]
    lemma = [char2id[c] for c in lemma]

    global character_lookup

    return [torch.cat([character_lookup(torch.tensor([c])), context], 1)
            for c in lemma]

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
    prevword = s[i-1][WF] if i > 0 else EOS
    prevlemma = s[i-1][LEMMA] if i > 0 else EOS
    prevmsd = s[i-1][MSD] if i > 0 else EOS
    nextword = s[i+1][WF] if i + 1 < len(s) else EOS
    lemma = s[i][LEMMA] 
    nextlemma = s[i+1][LEMMA] if i + 1 < len(s) else EOS
    nextmsd = s[i+1][MSD] if i + 1 < len(s) else EOS

    prevword = torch.tensor([dropitem(prevword,wf2id,training)])
    nextword = torch.tensor([dropitem(nextword,wf2id,training)])
    prevlemma = torch.tensor([dropitem(prevlemma,lemma2id,training)])
    nextlemma = torch.tensor([dropitem(nextlemma,lemma2id,training)])
    prevmsd = torch.tensor([dropitem(prevmsd,msd2id,training)])
    nextmsd = torch.tensor([dropitem(nextmsd,msd2id,training)])
    lemma = torch.tensor([dropitem(lemma,lemma2id,training)])

    return embed_context(prevword,prevlemma,prevmsd,lemma,nextword,nextlemma,nextmsd)

def get_loss(i, s, encoder_fwd, encoder_bwd, decoder):

    context = get_context(i,s,training=1)
    # dy.renew_cg()  
    # enc_fwd_lstm.set_dropout(LSTMDROPOUT)
    # enc_bwd_lstm.set_dropout(LSTMDROPOUT)
    # dec_lstm.set_dropout(LSTMDROPOUT)
    embedded = embed(s[i][LEMMA], context)
    encoded = encode(embedded, encoder_fwd, encoder_bwd)
    loss =  decode(encoded, s[i][WF], decoder)

    # enc_fwd_lstm.set_dropout(0)
    # enc_bwd_lstm.set_dropout(0)
    # dec_lstm.set_dropout(0)
    
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

def eval(devdata,id2char,encoder_fwd, encoder_bwd, decoder,generating=1, outf=None):
    input, gold = devdata
    corr = 0.0
    lev=0.0
    tot = 0.0
    for n,s in enumerate(input):
        for i,fields in enumerate(s):
            wf, lemma, msd = fields            
            if msd == NONE and lemma != NONE:
                if generating:
                    wf = generate(i,s,id2char, encoder_fwd, encoder_bwd, decoder)
                if wf == gold[n][i][WF]:
                    corr += 1
                lev += levenshtein(wf,gold[n][i][WF])[2]
                tot += 1
            if outf:
                outf.write('%s\n' % '\t'.join([wf,lemma,msd]))
        if outf:
            outf.write('\n')
    return (0,0) if tot == 0 else (corr / tot, lev/tot)

def generate(i, s, id2char, encoder_fwd, encoder_bwd, decoder):
    """ Generate a word form for the lemma at position i in sentence s. """
    context = get_context(i,s)
    embedded = embed(s[i][LEMMA],context)
    encoded = encode(embedded, encoder_fwd, encoder_bwd)

    in_seq = s[i][LEMMA]
    input_mat = torch.cat(encoded)
    w1dt = None

    last_output_embeddings = output_lookup(torch.tensor([char2id[EOS]]))
    initial_input = torch.cat((torch.zeros(STATE_SIZE*2).view(1,-1), last_output_embeddings), 1)
    _, decoder.state = decoder.lstm(initial_input.view(1, 1, -1), decoder.state)
    
    out = ''
    count_EOS = 0
    w1dt = decoder.w1(torch.t(input_mat))
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        # w1dt = w1dt or w1 * input_mat
        vector = torch.cat((decoder.attend(input_mat, w1dt), last_output_embeddings.view(-1,1)))
        out_decoder, decoder.state = decoder.lstm(vector.view(1, 1, -1), decoder.state)
        out_vector = decoder.linear(out_decoder)

        # probs = dy.softmax(out_vector).vec_value()
        probs = F.softmax(out_vector.view(1,-1)).view(-1)
        max_, idx = probs.max(0)
        next_char = idx.item()
        
        last_output_embeddings = output_lookup(torch.tensor([next_char]))
        if id2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += id2char[next_char]
    return out

def train(traindata,devdata,wf2id,lemma2id,char2id,id2char,msd2id,epochs=20):
    # trainer = dy.AdamTrainer(model)
    encoder_fwd = LSTMEncoder(LSTM_NUM_OF_LAYERS, 8*EMBEDDINGS_SIZE, STATE_SIZE)
    encoder_bwd = LSTMEncoder(LSTM_NUM_OF_LAYERS, 8*EMBEDDINGS_SIZE, STATE_SIZE)
    decoder = AttnLSTMDecoder(LSTM_NUM_OF_LAYERS,STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE)

    encoder_fwd_optimizer = optim.SGD(encoder_fwd.parameters(), lr=0.1)
    encoder_bwd_optimizer = optim.SGD(encoder_bwd.parameters(), lr=0.1)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        print("EPOCH %u" % (epoch + 1))
        shuffle(traindata)
        total_loss = 0
        for n,s in enumerate(traindata):
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                stdout.write("Example %u of %u\r" % 
                             (n+1,len(traindata)))
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                   and random() < SAMPLETRAIN:
                    loss = get_loss(i,s, encoder_fwd, encoder_bwd, decoder)
                    loss_value = loss.item()
                    loss.backward()
                    encoder_fwd_optimizer.step()
                    encoder_bwd_optimizer.step()
                    decoder_optimizer.step()
                    total_loss += loss_value
        print("\nLoss per sentence: %.3f" % (total_loss/len(traindata)))
        print("Example outputs:")
        for s in traindata[:5]:
            for i,fields in enumerate(s):
                wf, lemma, msd = fields
                if (iscandidatemsd(msd) or (msd == NONE and lemma != NONE))\
                   and random() < SAMPLETRAIN:
                    print("INPUT:", s[i][LEMMA], "OUTPUT:",
                          generate(i,s,id2char, encoder_fwd, encoder_bwd, decoder),
                          "GOLD:",wf)
                    break
        # devacc, devlev = eval(devdata,id2char,encoder_fwd, encoder_bwd, decoder)
        # print("Development set accuracy: %.2f" % (100*devacc))
        # print("Development set avg. Levenshtein distance: %.2f" % devlev)
        # print()


if __name__=='__main__':
    traindata, wf2id, lemma2id, char2id, msd2id = read_dataset(sysargv[1])
    devinputdata, _, _, _, _ = read_dataset(sysargv[2])
    devgolddata, _, _, _, _ = read_dataset(sysargv[3])

    id2char = {id:char for char,id in char2id.items()}
    init_model(wf2id,lemma2id,char2id,msd2id)
    train(traindata,[devinputdata,devgolddata],
          wf2id,lemma2id,char2id,id2char,msd2id,20)    
    # eval([devinputdata,devgolddata],id2char,generating=1,
    #      outf=open("%s-out" % sysargv[2],"w"))
