import numpy as np
import pandas as pd

#import gensim
#import random
#import re
#from collections import Counter
#from keras.utils.generic_utils import get_custom_objects
from keras import regularizers
#from keras import backend as K
#import pickle
#import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

### Parameters ###
RAWDATA_LABEL_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw4_RNN/training_label.txt'
OBSERVE_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw4_RNN/testing_data.txt'
SUBMISSION_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw4_RNN/sampleSubmission.csv'
MODE_PATH = './model'
HIS_PATH = './history'
##################

#stemmer = gensim.parsing.porter.PorterStemmer()
#def preprocess(string, use_stem = True):
#    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt").replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont").replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt").replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt").replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt").replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ").replace("couldn ' t","couldnt")
#    for same_char in re.findall(r'((\w)\2{2,})', string):
#        string = string.replace(same_char[0], same_char[1])
#    for digit in re.findall(r'\d+', string):
#        string = string.replace(digit, "1")
#    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
#        if punct[0:2] =="..":
#            string = string.replace(punct, "...")
#        else:
#            string = string.replace(punct, punct[0])
#    return string
#
#s = "been there ,,, still there .,... but he can ' t complain when he runs out of clean clothes i taught mine how to do his at 10"
#print("preprocess: ",preprocess(s))
#print("after stemming: ",stemmer.stem_sentence(preprocess(s)))

def Preprocessing(corpus,corpus_y,MODE,pad_length=None):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    ### Parameters ###
    BOW_MODE = 'binary' # default  
    ##################
    corpus_y = np.asarray(corpus_y,dtype='int32')
    if MODE == 'sequences':
        ## Data preprocessing (tokenize, padding)
        tokenizer = Tokenizer(num_words=None, 
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                              lower=True, 
                              split=' ', 
                              char_level=False, 
                              oov_token=None, 
                              document_count=0)
        tokenizer.fit_on_texts(corpus)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(corpus)
        sequences = pad_sequences(sequences, maxlen=pad_length) 
        return sequences, corpus_y, word_index
    elif MODE == 'bow':
        ## Data preprocessing (Bag Of Word)
        # To avoid the overloading of dimension, it could decress the dict size.
        word_index = tokenizer.word_index
        bow = tokenizer.texts_to_matrix(corpus,BOW_MODE)
        return bow, corpus_y, word_index

def Build_model(SEQUENCES_SIZE, LEXION_SIZE, HIDDEN_DIM):
    ## Build Simple RNN Model
    from keras.layers import Input,InputLayer,Embedding,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU,Activation
    from keras.models import Model
    ### Parameters ###
    RETURN_SEQUENCE = False
    DROPOUT_RATE = 0.4    
    ##################
    # Input
    inputs = Input(shape=(SEQUENCES_SIZE,),dtype='int32')
    # Embedding layer
    # This the important issue that input_dim: int > 0. 
    # Size of the vocabulary, i.e. maximum integer index + 1.
    embedding_inputs = Embedding(input_dim=LEXION_SIZE+1,
                                 output_dim=HIDDEN_DIM,
                                 trainable=True)(inputs)
    # LSTM
    RNN_cell = LSTM(HIDDEN_DIM,return_sequences=RETURN_SEQUENCE,dropout=DROPOUT_RATE)
    RNN_output = RNN_cell(embedding_inputs)
    
    # DNN (classify)
    outputs = Dense(2,activation='softmax',
                    kernel_regularizer=regularizers.l2(0.0001))(RNN_output)
    
    model = Model(inputs,outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.summary()
    return model

def Train_model(X,Y,X_val,Y_val,model,now):
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    ### Parameter ###
    PATIENCE = 5
    EPOCHS = 30
    BATCH_SIZE = 5120
    #################
    cp = ModelCheckpoint(MODE_PATH + '/model_'+ str(now) +'.h5',
                         verbose=1,save_best_only=True,monitor='val_acc')
    es = EarlyStopping(monitor='val_acc',patience=PATIENCE,verbose=1)
    history = model.fit(X,Y,
                        validation_data=(X_val,Y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[cp,es])
    return history

def ensemble(NUM):
    ## Read Raw Data
    with open(RAWDATA_LABEL_PATH, encoding = 'utf8') as f:
        raw_label = f.readlines()
    X_raw = [seg.strip().split(" +++$+++ ")[1] for seg in raw_label]
    y_raw = [seg.strip().split(" +++$+++ ")[0] for seg in raw_label]
    ## Preprocessing
    from keras.utils import to_categorical    
    X_raw, y_raw, word_index = Preprocessing(X_raw,y_raw,'sequences')
    y_raw = to_categorical(y_raw).astype('int32')
    ### Training Parameters ###
    TRAIN_SIZE = 0.7
    SEQUENCES_SIZE = np.shape(X_raw)[1] # columns dim size of sequences
    LEXION_SIZE = len(word_index)
    HIDDEN_DIM = 32
    ###########################
    ## Ensemble Training
    from sklearn.model_selection import train_test_split    
    for i in range(NUM):
        # Get Training set and Validation set
        X_train,X_val,y_train,y_val = train_test_split(X_raw,y_raw,train_size=TRAIN_SIZE,
                                               random_state=i,
                                               shuffle=True)
        # Build new model
        model = Build_model(SEQUENCES_SIZE,LEXION_SIZE,HIDDEN_DIM)
        # Train the model
        Train_model(X_train,y_train,X_val,y_val,model,i)
        
    ## Read Observe Data
    with open(OBSERVE_PATH, encoding = 'utf8') as f:
        ob = f.readlines()
    ob_id = [seg.strip().split(",",1)[0] for seg in ob][1:]
    ob = [seg.strip().split(",",1)[1] for seg in ob][1:]
    ## Preprocessing
    ob, ob_id, word_index = Preprocessing(ob, ob_id, 'sequences',SEQUENCES_SIZE)
        
    ### Predicting Parameters ###
    BATCH_SIZE = 5120
    #############################
    ## Ensemble Predicting
    pred = np.zeros((len(ob_id),2))
    from keras.models import load_model
    for i in range(NUM):
        model = load_model(MODE_PATH + '/model_'+ str(i) +'.h5')
        pred += model.predict(ob,batch_size=BATCH_SIZE,verbose=1)
    pred /= NUM
    
    ## Submission
    print("Submission ...")
    sampleSubmission = pd.read_csv(SUBMISSION_PATH)
#    sampleSubmission["label"] = np.round(pred)
    sampleSubmission["label"] = np.argmax(pred,axis=-1)
    sampleSubmission.to_csv(SUBMISSION_PATH,index=None)   
    
if __name__ == '__main__':
    ensemble(5)
