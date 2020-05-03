# sources:
# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/
# https://nextjournal.com/gkoehler/machine-translation-seq2seq-cpu
# https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py

from gensim.models import Word2Vec
import sys
import numpy
from numpy import array
from numpy import argmax
from numpy import dot
from numpy.linalg import norm
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense

MAX_SEQ_LEN = 0
TEXT_DIM = 100
POS_LIM = 60
STEP = 0

def main():
    
    PARAM_NUM = 5
    if len(sys.argv) < PARAM_NUM:
        print("Need " + str(PARAM_NUM-len(sys.argv)) + " more args")
        print("<vectorized words> <bi-text> <step> <model output>")
        return

    STEP = int(sys.argv[3])
    Word2VecModel = Word2Vec.load(sys.argv[1])
    print("Word2Vec model loaded")

    with open(sys.argv[2], 'r') as ct:
        lines = ct.readlines()
    print("Sentences loaded")

    text, pos, pos2num = split(lines)
    print("Max sequence length: " + str(MAX_SEQ_LEN))

    Model = seq2seq_model_builder(400)
    
    text_array, step_pos = vectorize(text[:STEP], pos[:STEP], Word2VecModel)
    pos_array = pos_vectorize(step_pos, pos2num)

    encoder_input_data = pos_array
    decoder_input_data = text_array[:,1:,:]
    decoder_target_data = text_array[:,:-1,:]

    Model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size = 10,
              epochs = 1,
              workers=4,
              validation_split=0.2)
    Model.save(sys.argv[4])
              

def split(lines):
    global MAX_SEQ_LEN
    count = 0
    error = 0
    text = []
    pos = []
    pos2num = dict()
    pos_num = 0
    
    for i in range(int(len(lines)/10000)):
        print("_", end="", flush=True)
    print("")

    for line in lines:
        count += 1
        if count % 10000 == 0:
            print(".", end="", flush=True)
        one_text = []
        one_pos = []
        temp = line.split("\t")[:-1]
        try:
            if temp.count("|") > 1:
                raise ValueError("Too many '|'")
            flag = True
            for word in temp:
                if word == "|":
                    flag = False
                    continue
                if word  == "\n" or word == "":
                    continue
                if flag:
                    one_text.append(word)
                else:
                    one_pos.append(word)
                    if not word in pos2num: # word is a pos tag
                        tmp = [0 for _ in range(0, POS_LIM)]
                        tmp[pos_num] = 1
                        pos2num[word] = array(tmp)
                        pos_num += 1
            if len(one_text) != len(one_pos):
                raise ValueError("Mismatch in line")
            if len(one_text) > MAX_SEQ_LEN:
                MAX_SEQ_LEN = len(one_text)
            text.append(one_text)
            pos.append(one_pos)
        except ValueError:
            error += 1
            continue


    if len(text) != len(pos):
        raise ValueError("Mismatch in lenght of text and pos")
        
    print("")
    print(str(len(text)) + " total lines split")
    print(str(error) + " total errors not included")

    return text, pos, pos2num

def vectorize(text, pos, Word2VecModel):
    text_array = []
    max_len = MAX_SEQ_LEN + 2
    dot_step = int(len(text)/70)
    step_pos = []
    count = 0
    error = 0
    for sentence, sen_pos in zip(text, pos):
        count += 1
        if count % dot_step == 0:
            print(".", end="", flush=True)
        t_array = numpy.zeros((max_len, TEXT_DIM), dtype="float32")
        flag = False
        t_array[0,:] = numpy.full((TEXT_DIM,), 5, dtype="float32")
        i = 0
        for word in sentence:
            i += 1
            if not word in Word2VecModel.wv:
                flag = True
                break
            else:
                t_array[i,:] = Word2VecModel.wv[word]
        if flag:
            error += 1
            break
        
        text_array.append(numpy.stack(t_array))
        step_pos.append(sen_pos)

    print("")

    if len(text_array) != len(step_pos):
        raise ValueError("Mismatch after text vectorization")

    print(str(count) + " text sentences vectorized")
    print(str(error) + " errors occured")

    return numpy.stack(text_array), step_pos

def pos_vectorize(pos, pos2num):
    pos_array = []
    step = len(pos)/70
    max_len = MAX_SEQ_LEN
    count = 0
    error = 0
    for sen_pos in pos:
        count += 1
        if count % step  == 0:
            print(".", end="", flush=True)
        p_arr = numpy.zeros((max_len, POS_LIM), dtype="float32")
        i = -1
        for p in sen_pos:
            i += 1
            p_arr[i,:] = pos2num[p]
        pos_array.append(numpy.stack(p_arr))

    print("")
    print(str(count) + " total lines pos_vectorized")
    print(str(error) + " errors")
        
    return pos_array
    
def seq2seq_model_builder(hidden_dim):
    global pos2num

    encoder_inputs = Input(shape=(MAX_SEQ_LEN, POS_LIM), dtype="float32")
    encoder_lstm = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(MAX_SEQ_LEN+1, TEXT_DIM), dtype="float32")
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(TEXT_DIM, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.summary()

    return model

def evalWE(y_true, y_pred):
    dist = numpy.zeros((y_true.shape,), dtype="float32")
    for i in range(0, len(y_true)):
        dist[i] = cosine(y_true[i,:], y_pred[i,:])
    return norm(dist)
        

def cosine(vec1, vec2):
    return dot(vec1, vec2)/(norm(vec1)*norm(vec2))

if __name__ == "__main__":
    main()
