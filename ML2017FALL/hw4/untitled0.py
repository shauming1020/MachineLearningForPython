import numpy as np
import pandas as pd
#import random
#from collections import Counter
#from keras.utils.generic_utils import get_custom_objects
#from keras import backend as K
#import pickle
#import sys

### Parameters ###
RAWDATA_LABEL_PATH = './dataset/training_label.txt'
OBSERVE_PATH = './dataset/testing_data.txt'
SUBMISSION_PATH = './dataset/sampleSubmission.csv'
MODE_PATH = './model'
HIS_PATH = './history'
##################

def Preprocess(string, use_stem = True):
    import re
    import gensim
    stemmer = gensim.parsing.porter.PorterStemmer()
    string = string.replace("i ' m", "im").replace("you ' re","youre")
    string = string.replace("didn ' t","didnt").replace("can ' t","cant")
    string = string.replace("haven ' t", "havent").replace("won ' t", "wont")
    string = string.replace("isn ' t","isnt").replace("don ' t", "dont")
    string = string.replace("doesn ' t", "doesnt").replace("aren ' t", "arent")
    string = string.replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")
    string = string.replace("ain ' t","aint").replace("shouldn ' t","shouldnt")
    string = string.replace("wasn ' t","wasnt").replace(" ' s","s")
    string = string.replace("wudn ' t","wouldnt").replace(" .. "," ... ")
    string = string.replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return stemmer.stem_sentence(string)

def Encoding(corpus,corpus_y,MODE,pad_length=None):
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

def PreTrain_weights(sequences, word_index, HIDDEN_DIM):
    from gensim.models import Word2Vec
    sequences = [seg.split(" ") for seg in sequences]
    w2v_model = Word2Vec(sequences, size=HIDDEN_DIM, window=5, min_count=0, workers=8)
#    w2v_model.build_vocab(sentences)
#    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
#    word_vectors = w2v_model.wv
#    del w2v_model
    SEQUENCES_SIZE = np.max([len(i) for i in sequences])
    EMB_INPUT_SIZE = len(word_index)
    embedding_matrix = np.zeros((SEQUENCES_SIZE,EMB_INPUT_SIZE))
    oov_count = 0
    for word, i in word_index.items():
        try:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector 
        except:
            oov_count +=1
            print(word," not in w2v model")
    return SEQUENCES_SIZE, EMB_INPUT_SIZE, embedding_matrix


def Build_model(SEQUENCES_SIZE, EMB_INPUT_SIZE, HIDDEN_DIM, embedding_matrix=None):
    ## Build Simple RNN Model
    from keras.layers import Input,InputLayer,Embedding,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU,Activation
    from keras.models import Model
    from keras import regularizers
    ### Parameters ###
    RETURN_SEQUENCE = False
    DROPOUT_RATE = 0.4
    ##################
    # Input
    inputs = Input(shape=(SEQUENCES_SIZE,),dtype='int32')  
    # Embedding layer
    # This the important issue that input_dim: int > 0. 
    # Size of the vocabulary, i.e. maximum integer index + 1.    
    if embedding_matrix.any():

        embedding_inputs = Embedding(input_dim=EMB_INPUT_SIZE+1,
                                     output_dim=HIDDEN_DIM,
                                     weights=[embedding_matrix],
                                     trainable=False)(inputs)        
    else:
        embedding_inputs = Embedding(input_dim=EMB_INPUT_SIZE+1,
                                     output_dim=HIDDEN_DIM,
                                     trainable=True)(inputs)
    # LSTM
    RNN_cell = LSTM(HIDDEN_DIM,return_sequences=RETURN_SEQUENCE,dropout=DROPOUT_RATE)
    RNN_output = RNN_cell(embedding_inputs)
    
    # DNN (classify)
    outputs = Dense(2,activation='softmax',
                    kernel_regularizer=regularizers.l2(0.01))(RNN_output)
    
    model = Model(inputs,outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.summary()
    return model

def Train_model(X,Y,X_val,Y_val,model,now):
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    ### Parameter ###
    PATIENCE = 8
    EPOCHS = 32
    BATCH_SIZE = 5120
    #################
    cp = ModelCheckpoint(MODE_PATH + '/model_' + str(now) + '.h5',
                         verbose=1,save_best_only=True,monitor='val_acc')
    es = EarlyStopping(monitor='val_acc',patience=PATIENCE,verbose=1)
    history = model.fit(X,Y,
                        validation_data=(X_val,Y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[cp,es])
    return history

def Stacked_dataset(members, X_test):
    # Prepare a training dataset for the meta-learner
    stackX = None
    for model in members:
        y_hat = model.predict(X_test,verbose=0)
        # stack predictions into [rows,members,probabilities]
        if stackX is None:
            stackX = y_hat
        else:
            stackX = np.dstack((stackX,y_hat))
    # flatten predictions to [rows,members,probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return stackX

def Fit_stacked_model(members,X_test,y_test):
    stackedX = Stacked_dataset(members,X_test)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(stackedX, y_test)
    return model

def Stacked_prediction(members,model,ob):
    stackedX = Stacked_dataset(members,ob)
    y_hat = model.predict(stackedX)
    return y_hat
    
def Stacked_Ensemble(NUM):
    ## Read Raw Data
    with open(RAWDATA_LABEL_PATH, encoding = 'utf8') as f:
        raw_label = f.readlines()
    X_raw = [seg.strip().split(" +++$+++ ")[1] for seg in raw_label]
    y_raw = [seg.strip().split(" +++$+++ ")[0] for seg in raw_label]
    ## Preprocessing
    X_raw_seq = [Preprocess(seg) for seg in X_raw]
    ## Encoding
    from keras.utils import to_categorical    
    X_raw, y_raw, word_index = Encoding(X_raw_seq,y_raw,'sequences')
    ## Get Training set and Testing set
    from sklearn.model_selection import train_test_split   
    X_train,X_test,y_train,y_test = train_test_split(X_raw,y_raw,train_size=0.7,shuffle=True)    
    ### Training Parameters ###
    VAL_SIZE = 0.2
    SEQUENCES_SIZE = np.shape(X_raw)[1] # columns dim size of sequences
    LEXION_SIZE = len(word_index)
    HIDDEN_DIM = 128
    ###########################
    ## Using the Pre-trained Word-Vector
    EMB_SEQ_SIZE, EMB_INPUT_SIZE, embedding_matrix = PreTrain_weights(X_raw_seq, word_index, HIDDEN_DIM)
    X_raw, y_raw, word_index = Encoding(X_raw_seq,y_raw,'sequences',EMB_SEQ_SIZE)
    X_train,X_test,y_train,y_test = train_test_split(X_raw,y_raw,train_size=0.7,shuffle=True) 
    
    ## Train and Save Sub-Models
    y_raw = to_categorical(y_raw).astype('int32')    
    for i in range(NUM):
        # Get Training set and Validation set
        X,X_val,y,y_val = train_test_split(X_train,y_train,
                                           test_size=VAL_SIZE,
                                           random_state=i,
                                           shuffle=True)
        # Build new model
#        model = Build_model(SEQUENCES_SIZE,LEXION_SIZE,HIDDEN_DIM)
        model = Build_model(EMB_SEQ_SIZE,EMB_INPUT_SIZE,HIDDEN_DIM,embedding_matrix)
        # Train the model
        Train_model(X,y,X_val,y_val,model,i)
        
    ## Separate Stacking Model
    from keras.models import load_model
    all_models = list()
    for i in range(NUM):
        model = load_model(MODE_PATH + '/model_'+ str(i) +'.h5')
        all_models.append(model)   
    yc_test = np.argmax(y_test,axis=-1) # LogisticRegression's target values    
    model = Fit_stacked_model(all_models,X_test,yc_test)
    ## Evaluate The Model
    from sklearn.metrics import accuracy_score
    y_pred = Stacked_prediction(all_models, model, X_test)
    acc = accuracy_score(yc_test, y_pred)
    print('Stacked Test Accuracy: %.3f' % acc)
    
    ## Read Observe Data
    with open(OBSERVE_PATH, encoding = 'utf8') as f:
        ob = f.readlines()
    ob_id = [seg.strip().split(",",1)[0] for seg in ob][1:]
    ob = [seg.strip().split(",",1)[1] for seg in ob][1:]
    ## Preprocessing
    ob = [Preprocess(seg) for seg in ob]    
    ## Encoding
    ob, ob_id, word_index = Encoding(ob, ob_id, 'sequences', SEQUENCES_SIZE)   
    ## Submission
    print("Submission ...")
    pred = Stacked_prediction(all_models,model,ob)
    sampleSubmission = pd.read_csv(SUBMISSION_PATH)
    sampleSubmission["label"] = np.argmax(pred,axis=-1)
    sampleSubmission.to_csv(SUBMISSION_PATH,index=None)   

if __name__ == '__main__':
    Stacked_Ensemble(5)
