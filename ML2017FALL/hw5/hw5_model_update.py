import pandas as pd
import numpy as np
import keras.backend as K 
import matplotlib.pyplot as plt
import os
import time

# only show error
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 

RAW_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/train.csv'
OBSERVE_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/test.csv'
SUBMIS_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/SampleSubmisson.csv'
MODEL = 'NNMF' # MF:Matrix Factorisation, NNMF:Non-negative Matrix Factorisation, DNN:Deep Neural Network
MODEL_DIR = './model'
MODEL_NAME = MODEL+'-00014-0.00394.h5'
MODEL_PATH = MODEL_DIR + '/' + MODEL_NAME
HIS_DIR = './history'
HIS_PATH = HIS_DIR + '/NNMF0.029789_history.npz'
PIC_DIR = './picture'

def read_data(path):
    print('=== Peak into the dataset ===')
    dataset = pd.read_csv(path)
    dataset.head()
    print('Assign a unique number between(0,#users/movies) to each user and movies.')
    dataset.UserID = dataset.UserID.astype('category').cat.codes.values
    dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values
    dataset.head()
    n_users, n_movies = len(dataset.UserID.unique()), len(dataset.MovieID.unique())
    return dataset, n_users, n_movies

def Do_normalize(x, do):
    if do == 'scale':
        return (x-1)/5
    elif do == 'std':
        return (x - np.mean(x))/np.std(x)
    elif do == 'mMfs':
        return (x - np.min(x)) / (np.max(x) - np.min(x))

# The function of rmse is for tensor.
def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def Building_new_model(MODEL,n_users, n_movies):
    import keras
    from keras.optimizers import Adam
#    from IPython.display import SVG
#    from keras.utils.vis_utils import model_to_dot    
    ###### Parameters ######
    N_LATENT_FACTORS = 50
    WORD_LENGTH = 1 # an UserID is a value to see as a word for embedding layer.
    OPT = 'adam'
    REGULAR_LMBDA = 0.00001
    ########################
    if MODEL == 'MF':
        print('=== Matrix Factorisation in Keras ===')
        print('Build the matrix factorisation model.')
        movie_input = keras.layers.Input(shape=[WORD_LENGTH], name='Movie')
        movie_embedding = keras.layers.Embedding(n_movies+1, N_LATENT_FACTORS, name='Movie-Embedding')(movie_input)
        movie_drop = keras.layers.Dropout(0.1)(movie_embedding)
        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_drop)
    
        user_input = keras.layers.Input(shape=[WORD_LENGTH], name='User')
        user_embedding = keras.layers.Embedding(n_users+1, N_LATENT_FACTORS, name='User-Embedding')(user_input)
        user_drop = keras.layers.Dropout(0.1)(user_embedding)
        user_vec = keras.layers.Flatten(name='FlattenUsers')(user_drop)
        
        prod = keras.layers.Dot(axes=1,name='DotProduct')([movie_vec, user_vec])
        
        movie_bias_embedding = keras.layers.Embedding(n_movies+1, 1, embeddings_initializer='zeros', name='Movie_bias-Embedding')(movie_input)
        movie_bias_drop = keras.layers.Dropout(0.01)(movie_bias_embedding)
        movie_bias_vec = keras.layers.Flatten(name='FlattenMoviesBias')(movie_bias_drop)
        
        user_bias_embedding = keras.layers.Embedding(n_users+1, 1, embeddings_initializer='zeros', name='User_bias-Embedding')(user_input)
        user_bias_drop = keras.layers.Dropout(0.01)(user_bias_embedding)
        user_bias_vec = keras.layers.Flatten(name='FlattenUsersBias')(user_bias_drop)
        
        add = keras.layers.Add(name='Add')([movie_bias_vec, user_bias_vec, prod])
        add_drop = keras.layers.Dropout(0.2)(add)
        model = keras.Model([user_input, movie_input], add_drop)
             
    if MODEL == 'NNMF':
        print('=== Non-negative Matrix Factorisation(NNMF) in keras ===')
        print('Build the NNMF model.')
        from keras.constraints import non_neg
        from keras.regularizers import l2
        movie_input = keras.layers.Input(shape=[WORD_LENGTH],name='Movie')
        movie_embedding = keras.layers.Embedding(n_movies+1, 
                                                 N_LATENT_FACTORS, 
                                                 name='NonNegMovie-Embedding', 
                                                 embeddings_constraint=non_neg(),
                                                 embeddings_regularizer=l2(REGULAR_LMBDA))(movie_input)
        movie_drop = keras.layers.Dropout(0.1,name='Movie-Dropout')(movie_embedding)
        
        movie_vec = keras.layers.Flatten(name='FlattemMovies')(movie_drop)
        
        user_input = keras.layers.Input(shape=[WORD_LENGTH],name='User')
        user_embedding = keras.layers.Embedding(n_users+1,
                                                N_LATENT_FACTORS,
                                                name='NonNegUser-Embedding',
                                                embeddings_constraint=non_neg(),
                                                embeddings_regularizer=l2(REGULAR_LMBDA))(user_input)
        user_drop = keras.layers.Dropout(0.1,name='User-Dropout')(user_embedding)
        user_vec = keras.layers.Flatten(name='FlattenUsers')(user_drop)
        
        prod = keras.layers.Dot(axes=1,name='DotProduct')([movie_vec, user_vec])

        # Do bias embedding need to be constraints?
        movie_bias_embedding = keras.layers.Embedding(n_movies+1, 1, 
                                                      embeddings_initializer='zeros', 
                                                      name='Movie_bias-Embedding', 
                                                      embeddings_constraint=non_neg(),
                                                      embeddings_regularizer=l2(REGULAR_LMBDA))(movie_input)
        movie_bias_drop = keras.layers.Dropout(0.1,name='Movie_bias-Dropout')(movie_bias_embedding)
        movie_bias_vec = keras.layers.Flatten(name='FlattenMoviesBias')(movie_bias_drop)

        user_bias_embedding = keras.layers.Embedding(n_users+1, 1, 
                                                     embeddings_initializer='zeros', 
                                                     name='User_bias-Embedding',
                                                     embeddings_constraint=non_neg(),
                                                     embeddings_regularizer=l2(REGULAR_LMBDA))(user_input)
        user_bias_drop = keras.layers.Dropout(0.1,name='User_bias-Dropout')(user_bias_embedding)
        user_bias_vec = keras.layers.Flatten(name='FlattenUsersBias')(user_bias_drop)
        
        add = keras.layers.Add(name='Add')([movie_bias_vec, user_bias_vec, prod])
        add_drop = keras.layers.Dropout(0.1,name='Add-Dropout')(add)
        
        model = keras.Model([user_input, movie_input], add_drop)
        
    if MODEL == 'DNN':
        print('=== Nerual networks for recommendation ===' )
        N_LATENT_FACTORS_USER = 5
        N_LATENT_FACTORS_MOVIE = 8
#        SEED = 4 # if we define a random seed for dropout, then we will get a different number 
#        after each random in dropout.
        movie_input = keras.layers.Input(shape=[WORD_LENGTH],name='Movie')
        movie_embedding = keras.layers.Embedding(n_movies+1,
                                                 N_LATENT_FACTORS_MOVIE, 
                                                 name='Movie-Embedding')(movie_input)
        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)
        
        user_input = keras.layers.Input(shape=[WORD_LENGTH], name='User')
        user_embedding = keras.layers.Embedding(n_users+1,
                                                 N_LATENT_FACTORS_USER, 
                                                 name='User-Embedding')(user_input)
        user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
        user_vec = keras.layers.Dropout(0.2)(user_vec)
        
        concat = keras.layers.Concatenate(name='Concat')([movie_vec, user_vec])
        concat_dropout = keras.layers.Dropout(0.2)(concat)
        dense = keras.layers.Dense(256, name='FullyConnected')(concat_dropout)
        dropout_1 = keras.layers.Dropout(0.2,name='Dropout_1')(dense)
        dense_2 = keras.layers.Dense(128,name='FullyConnected-1')(dropout_1)
        dropout_2 = keras.layers.Dropout(0.2,name='Dropout_2')(dense_2)
        dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dropout_2)
        dropout_3 = keras.layers.Dropout(0.2,name='Dropout_3')(dense_3)
        dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dropout_3)
        
        result = keras.layers.Dense(1, activation='softmax',name='Activation')(dense_4)
        OPT = Adam(lr=0.005)
        model = keras.Model([user_input, movie_input], result)
        
    print('model compile...')
    print('loss: mse, acc: binary, early_stopping loss: rmse') 
    model.compile(optimizer=OPT,loss= 'mean_absolute_error',metrics=['accuracy',rmse])
    #SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
    model.summary()
    print('save the model...')
    model.save(MODEL_DIR+ '/' + MODEL+ '-00000-0.h5')
    return model

def Training_model(train_dataset, model):
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    ################ Parameters ################
    BATCH_SIZE = 1024
    EPOCHS = 100    
    VAL_SPLIT = 0.3
    PATIENCE = 30
    print('Number of UserID :',len(train_dataset.UserID.unique()), 
      ',Number of MovieID : ',len(train_dataset.MovieID.unique()))
    ############################################ 
    print('==================== Start Traning... ====================')
    # Shuffle training data
    train_dataset = train_dataset.sample(frac=1)
    # 'rmse' as a monitor to check the current loss, instead of 'mse' .
    cp = ModelCheckpoint(MODEL_DIR+ '/' + MODEL+"-{epoch:05d}-{val_acc:.5f}.h5",
                         monitor='val_acc', save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_rmse', patience=PATIENCE, mode='min')
    history = model.fit([train_dataset.UserID, train_dataset.MovieID], train_dataset.Rating, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=VAL_SPLIT,
                        verbose=1, callbacks=[cp,es])
    print('=== Save History... ===')
    H = history.history
    best_val = '{:.6f}'.format(np.max(H['val_acc']))
    last_val = '{:.6f}'.format(H['val_acc'][-1])
    model_name = MODEL + '-' +best_val + '_history.npz'
    print('Best val: ' + best_val)
    print('Last val: ' + last_val)       
    np.savez(HIS_DIR + '/' + model_name ,
             acc=H['acc'], val_acc=H['val_acc'],
             loss=H['loss'], val_loss=H['val_loss'],
             rmse=H['rmse'], val_rmse=H['val_rmse'])
    print('==================== END TRAINING ====================')
    return H, model_name
    
def Plot_history(H,model_name):   
    print('=== Plot training & val accuracy values ===' )
    plt.clf()
    acc, val_acc = H['acc'], H['val_acc']
    plt.plot(acc,'b'), plt.plot(val_acc,'r')
    plt.xlabel("Epoch"), plt.ylabel("Accuracy")    
    plt.legend(['acc','val_acc'], loc='upper left')  
    plt.title("Training Process of " + model_name)
    plt.savefig(PIC_DIR+ '/' + model_name + '_acc_history.png')
    plt.show()
    plt.close()    
    
    print('=== Plot training & val loss values ===')
    plt.clf()
    loss, val_loss = H['loss'], H['val_loss']   
    plt.plot(loss,'b'), plt.plot(val_loss,'r')
    plt.xlabel("Epoch"), plt.ylabel("Loss")
    plt.legend(['loss','val_loss'], loc='upper left')
    plt.title("Training Process of " + model_name)
    plt.savefig(PIC_DIR+ '/' + model_name + '_loss_history.png')
    plt.show()
    plt.close()    

    print('=== Plot training & val rmse values ===')
    plt.clf()
    rmse, val_rmse = H['rmse'], H['val_rmse']   
    plt.plot(rmse,'b'), plt.plot(val_rmse,'r')
    plt.xlabel("Epoch"), plt.ylabel("RMSE")
    plt.legend(['rmse','val_rmse'], loc='upper left')
    plt.title("Training Process of " + model_name)
    plt.savefig(PIC_DIR+ '/' + model_name + '_rmse_history.png')
    plt.show()
    plt.close()    
    
def Predict_and_submission(test_dataset, submission, model):
    print('=== Predicting ... ===')
    pred_rating = model.predict([test_dataset.UserID, test_dataset.MovieID])
    pred_rating_hat = np.round(pred_rating, 0) 
    print('=== Output the sample submission===')
    sampleSubmission = pd.read_csv(submission)
    sampleSubmission["Rating"] = pred_rating_hat
    sampleSubmission.to_csv(submission,index=None)    
    return pred_rating_hat

def Extracting_learnt_embedding(args,model):
    print('=== Extracting the learnt embeddings ===')
    for em_name in args:
        movie_embedding_learnt = model.get_layer(name=em_name).get_weights()[0]
        pd.DataFrame(movie_embedding_learnt).describe()

def Train_test_split(dataset):
    print('=== Split the dataset to train & test data ===')
    TEST_SIZE = 0.3
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dataset, test_size=TEST_SIZE)
    return train, test

def Evaluate_model(test_dataset, model):
    print('=== Evaluating... ===')
    BATCH_SIZE = 1024
    loss, acc, rmse = model.evaluate([test_dataset.UserID, test_dataset.MovieID], 
                                       test_dataset.Rating, 
                                       batch_size=BATCH_SIZE)
    print('loss: %.5f'%loss, 'acc: %.5f'%acc, 'rmse: %.5f'%rmse)
    return np.round(loss,5), np.round(acc,5), np.round(rmse,5)

def main():
    ###### Read raw dataset ######
    raw_dataset, n_users, n_movies = read_data(RAW_PATH)
    ###### Normalize ######
#    raw_dataset.Rating = Do_normalize(raw_dataset.Rating,'std')
#    raw_dataset.head()
    
    ###### Building the model ######
    model = Building_new_model(MODEL,n_users, n_movies)
    ###### Loading the model ######
#    from keras.models import load_model
#    model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})    
    
    ###### Split testing data from training data ######
    train_dataset, test_dataset = Train_test_split(raw_dataset)
        
    t_start = time.time()
    ###### Training model ######
    history, model_name = Training_model(train_dataset, model)
    print('=== Spent %s seconds ===' % np.round((time.time() - t_start), 3))
    
    ###### Evaluate the model ######
    Evaluate_model(test_dataset, model)
    
    ###### Read observing dataset ######
    ob_dataset, n_users_ob, n_movies_ob = read_data(OBSERVE_PATH)
    ###### Predicting the observing data ######
    Predict_and_submission(ob_dataset, SUBMIS_PATH, model)

    ###### Plot the history ######
#    history = np.load(HIS_PATH)
    Plot_history(history,model_name)

    ###### Extracting learnt embedding ###### 
#    embedding_layer_names = {'Movie-Embedding','User-Embedding'}
#    Extracting_learnt_embedding(embedding_layer_names, model)
    
if __name__ == '__main__':
#    args = parse_args()
#    main(args) 
    main()
