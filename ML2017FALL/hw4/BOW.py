# BOW MODEL
# Issue: Out Of Memory
import numpy as np
import pandas as pd

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

def Create_vocab(corpus,VOCAB_SIZE,save=False):
    from keras.preprocessing.text import Tokenizer    
     ## Data preprocessing (tokenize, padding)
#    VOCAB_SIZE  = 1000 # the maximum number of words to keep, based on word frequency.
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, 
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    if save:
        import pickle
        with open('tokenizer.pickle','wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer, word_index
    
def Encoding(corpus,tokenizer,pad_length=None):
    from keras.preprocessing.sequence import pad_sequences
    BOW_MODE = 'count' # default  
    ## Data preprocessing (Bag Of Word)
    # To avoid the overloading of dimension, it could decress the dict size.
    bow = tokenizer.texts_to_matrix(corpus,BOW_MODE).astype('int32')
    return bow

def Build_model(SEQUENCES_SIZE):
    ## Build Simple RNN Model
    from keras.layers import Input,InputLayer,Embedding,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU,Activation
    from keras.models import Model
    from keras import regularizers
    ### Parameters ###
    
    ##################
    # Input
    inputs = Input(shape=(SEQUENCES_SIZE,))  
    # DNN (classify)
    d1 = Dense(256,activation='relu',
               kernel_regularizer=regularizers.l2(0.01))(inputs)
    d1_drop = Dropout(0.4)(d1)
    outputs = Dense(2,activation='softmax',
                    kernel_regularizer=regularizers.l2(0.01))(d1_drop)
    outputs_drop = Dropout(0.4)(outputs)
    model = Model(inputs,outputs_drop)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    model.summary()
    return model

def Train_model(X,Y,X_val,Y_val,model,now):
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    ### Parameter ###
    PATIENCE = 8
    EPOCHS = 24
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
        y_hat = np.argmax(y_hat,axis=-1)
        # stack predictions into [rows,members,probabilities]
        if stackX is None:
            stackX = y_hat
        else:
            stackX = np.vstack((stackX,y_hat)) 
    # flatten predictions to [rows,members,probabilities]
    return stackX.transpose()

def Fit_stacked_model(members,X_test,yc_test):
    stackedX = Stacked_dataset(members,X_test)
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    clf1 = LogisticRegression(solver='lbfgs',multi_class='multinomial',
                              random_state=1)
#    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
#    clf3 = GaussianNB()
    model = VotingClassifier(estimators=[('rg',clf1)])
    model.fit(stackedX, yc_test)
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
    from keras.utils import to_categorical  
    X_raw_seq = [Preprocess(seg) for seg in X_raw]
    y_raw = to_categorical(y_raw).astype('int32') 
    ## Create the Vocab
    tokenizer, word_index = Create_vocab(X_raw_seq,5000,save=True)
    ## Encoding  
    X_raw = Encoding(X_raw_seq,tokenizer)   
    ## Get Training set and Testing set
    from sklearn.model_selection import train_test_split   
    X_train,X_test,y_train,y_test = train_test_split(X_raw,y_raw,train_size=0.7,shuffle=False)    
    ### Training Parameters ###
    VAL_SIZE = 0.2
    SEQUENCES_SIZE = np.shape(X_raw)[1] # columns dim size of sequences
    ###########################
 
    ## Train and Save Sub-Models  
    for i in range(NUM):
        # Get Training set and Validation set
        X,X_val,y,y_val = train_test_split(X_train,y_train,
                                           test_size=VAL_SIZE,
                                           random_state=i,
                                           shuffle=True)
        # Build new model
        model = Build_model(SEQUENCES_SIZE)
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
    ob_id = np.asarray(ob_id,dtype='int32')
    ob = [Preprocess(seg) for seg in ob]  
    ## Read the Vocab
    import pickle
    with open('tokenizer_vocab.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    ## Encoding
    ob = Encoding(ob, tokenizer) 
    ## Submission
    print("Submission ...")
    pred = Stacked_prediction(all_models,model,ob)
    sampleSubmission = pd.read_csv(SUBMISSION_PATH)
    sampleSubmission["label"] = np.argmax(pred,axis=-1)
    sampleSubmission.to_csv(SUBMISSION_PATH,index=None)   

if __name__ == '__main__':
    Stacked_Ensemble(5) 