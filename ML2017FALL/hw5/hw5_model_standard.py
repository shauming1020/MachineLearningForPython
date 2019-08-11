import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# only show error
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' 
RAW_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/train.csv'
OBSERVE_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/test.csv'
SUBMIS_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/SampleSubmisson.csv'

USER_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/users.csv'
ITEM_PATH = 'C:/Users/Zheng Shau Ming/Documents/GitHub/ML2017FALL_dataset/hw5_MF/movies.csv'

MODEL = 'NNMF' # MF:Matrix Factorisation, NNMF:Non-negative Matrix Factorisation, DNN:Deep Neural Network 
MODEL_DIR = './model'
MODEL_NAME = MODEL+'-00014-0.00394.h5'
MODEL_PATH = MODEL_DIR + '/' + MODEL_NAME
HIS_DIR = './history'
HIS_PATH = HIS_DIR + '/NNMF0.029789_history.npz'
PIC_DIR = './picture'
  
def Train_test_split(dataset):
    print('=== Split the dataset to train & test data ===')
    TEST_SIZE = 0.1
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dataset, test_size=TEST_SIZE)
    return train, test

def Do_normalize(x, do):
    if do == 'scale':
        return (x-1)/5
    elif do == 'std':
        return (x - np.mean(x))/np.std(x)
    elif do == 'mMfs':
        return (x - np.min(x)) / (np.max(x) - np.min(x))

# The function of rmse is for tensor.
def rmse(y_true, y_pred):
    import keras.backend as K 
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def Building_new_model(MODEL,n_users, n_movies, bias=False,n_latent_factors=16,m_latent_factors=16):
    import keras
    from keras.optimizers import Adam
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot    
    #### Parameters ####
    WORD_LENGTH = 1 # an UserID is a value to see as a word for embedding layer.
    OPT = 'adam'
    ####################
    if MODEL == 'MF':
        print('=== Matrix Factorisation in Keras ===')
        print('Build the matrix factorisation model.')
        movie_input = keras.layers.Input(shape=[WORD_LENGTH], name='Movie')
        movie_embedding = keras.layers.Embedding(n_movies, n_latent_factors, name='Movie-Embedding')(movie_input)
        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
        user_input = keras.layers.Input(shape=[WORD_LENGTH], name='User')
        user_embedding = keras.layers.Embedding(n_users, n_latent_factors, name='User-Embedding')(user_input)
        user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
        prod = keras.layers.Dot(axes=1,name='DotProduct')([movie_vec, user_vec])
        add = prod
        if bias:    
            movie_bias_embedding = keras.layers.Embedding(n_movies, 1, embeddings_initializer='zeros', name='Movie_bias-Embedding')(movie_input)
            movie_bias_vec = keras.layers.Flatten(name='FlattenMoviesBias')(movie_bias_embedding)      
            user_bias_embedding = keras.layers.Embedding(n_users, 1, embeddings_initializer='zeros', name='User_bias-Embedding')(user_input)
            user_bias_vec = keras.layers.Flatten(name='FlattenUsersBias')(user_bias_embedding)       
            add = keras.layers.Add(name='Add')([movie_bias_vec, user_bias_vec, prod])
        add_drop = keras.layers.Dropout(0.2)(add)
        model = keras.Model([user_input, movie_input], add_drop)
             
    if MODEL == 'NNMF':
        print('=== Non-negative Matrix Factorisation(NNMF) in keras ===')
        print('Build the NNMF model.')
        from keras.constraints import non_neg
        movie_input = keras.layers.Input(shape=[WORD_LENGTH],name='Movie')
        movie_embedding = keras.layers.Embedding(n_movies, 
                                                 n_latent_factors, 
                                                 name='NonNegMovie-Embedding', 
                                                 embeddings_constraint=non_neg())(movie_input)
        movie_vec = keras.layers.Flatten(name='FlattemMovies')(movie_embedding)     
        user_input = keras.layers.Input(shape=[WORD_LENGTH],name='User')
        user_embedding = keras.layers.Embedding(n_users,
                                                n_latent_factors,
                                                name='NonNegUser-Embedding',
                                                embeddings_constraint=non_neg())(user_input)
        user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)      
        prod = keras.layers.Dot(axes=1,name='DotProduct')([movie_vec, user_vec])
        add = prod
        if bias:
            # Do bias embedding need to be constraints?
            movie_bias_embedding = keras.layers.Embedding(n_movies, 1, 
                                                          embeddings_initializer='zeros', 
                                                          name='Movie_bias-Embedding', 
                                                          embeddings_constraint=non_neg())(movie_input)
            movie_bias_vec = keras.layers.Flatten(name='FlattenMoviesBias')(movie_bias_embedding)
            user_bias_embedding = keras.layers.Embedding(n_users, 1, 
                                                         embeddings_initializer='zeros', 
                                                         name='User_bias-Embedding',
                                                         embeddings_constraint=non_neg())(user_input)
            user_bias_vec = keras.layers.Flatten(name='FlattenUsersBias')(user_bias_embedding)    
            add = keras.layers.Add(name='Add')([movie_bias_vec, user_bias_vec, prod])            
        model = keras.Model([user_input, movie_input], add)
        
    if MODEL == 'DNN':
        print('=== Nerual networks for recommendation ===' )
#        SEED = 4 # if we define a random seed for dropout, then we will get a different number 
#        after each random in dropout.
        movie_input = keras.layers.Input(shape=[WORD_LENGTH],name='Movie')
        movie_embedding = keras.layers.Embedding(n_movies,
                                                 m_latent_factors, 
                                                 name='Movie-Embedding')(movie_input)
        movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
        movie_vec = keras.layers.Dropout(0.2)(movie_vec)
        
        user_input = keras.layers.Input(shape=[WORD_LENGTH], name='User')
        user_embedding = keras.layers.Embedding(n_users,
                                                 n_latent_factors, 
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
        model = keras.Model([user_input, movie_input], result)
        
    print('model compile...')
    print('loss: mse, acc: binary, early_stopping loss: rmse') 
    model.compile(optimizer=OPT,loss= 'mean_absolute_error',metrics=['accuracy',rmse])
    SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
    model.summary()
    print('save the model...')
    model.save(MODEL_DIR+ '/' + MODEL+ '-00000-0.h5')
    return model

def Training_model(train_dataset, model):
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    ################ Parameters ################
    BATCH_SIZE = 1024
    EPOCHS = 100    
    VAL_SPLIT = 0.05
    PATIENCE = 20
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
    name = [['acc','val_acc','Accuracy'],['loss','val_loss','Loss'],['rmse','val_rmse','RMSE']]
    print('=== Plot training & validating {acc,loss,rmse} values ===' )
    for i, j, k in name:
        see, val_see = H[i], H[j]
        plt.plot(see,'b'), plt.plot(val_see,'r')
        plt.xlabel("Epoch"), plt.ylabel(k)    
        plt.legend([i,j], loc='upper left')  
        plt.title("Training Process of " + model_name)
        plt.savefig(PIC_DIR+ '/' + model_name + '_'+ i +'_history.png')
        plt.show()
        plt.close()         

def Evaluate_model(test_dataset, model):
    print('=== Evaluating... ===')
    BATCH_SIZE = 1024
    loss, acc, rmse = model.evaluate([test_dataset.UserID, test_dataset.MovieID], 
                                       test_dataset.Rating, 
                                       batch_size=BATCH_SIZE)
    print('loss: %.5f'%loss, 'acc: %.5f'%acc, 'rmse: %.5f'%rmse)
    return np.round(loss,5), np.round(acc,5), np.round(rmse,5)
 
def Predict_and_submission(test_dataset, submission, model):
    print('=== Predicting ... ===')
    pred_rating = model.predict([test_dataset.UserID, test_dataset.MovieID])
    pred_rating_hat = np.round(pred_rating, 0) 
    print('=== Output the sample submission===')
    sampleSubmission = pd.read_csv(submission)
    sampleSubmission["Rating"] = pred_rating_hat
    sampleSubmission.to_csv(submission,index=None)    
    return pred_rating_hat


def Preprocess_data(dataset,UserInfo,ItemInfo):
    user_info = pd.read_csv(UserInfo, sep="::")
    user_id = dataset.UserID
    
    df = pd.DataFrame(columns=['Gender','Age'])
    for i in range(len(user_id)):
        id = user_id[i]
        print("now deal with i/total ",i,"/",len(user_id))
        info = user_info.loc[user_info["UserID"]==id, ("Gender","Age")]
        df = pd.concat([df,info],ignore_index=True)
    dataset['User_Gender'] = df.Gender
    dataset['User_Age'] = df.Age
    dataset.to_pickle('./preprocessing.pkl')    
#    df = pd.DataFrame(columns=['Gender','Age'])
#    j = 0
#    for i in range(len(user_info)):
#        id = user_info["UserID"][i]
#        print("now deal with i/total ",i,"/",len(user_id))
#        while(id == user_id[j]):
#            info = user_info.loc[[i],("Gender","Age")]
#            df = pd.concat([df,info],ignore_index=True)
#            j = j+1
#    dataset['User_Gender'] = df.Gender
#    dataset['User_Age'] = df.Age             
    
    movie_info = pd.read_csv(ItemInfo, sep="::")
    movie_info["Genres"] = movie_info["Genres"].str.split('|')
    movie_id = dataset.MovieID
    y = []
    for i in range(len(movie_id)):
        print("now deal with i/total ",i,"/",len(movie_id))
        id = movie_id[i]
        genres = movie_info.loc[movie_info["movieID"]==id,"Genres"].tolist()
        y+=genres
    dataset['Movie_Genres'] = y
    
    print('Assign a unique number between(0,#) to each user and movies.')
    dataset.Movie_Genres = dataset.Movie_Genres.astype('category').cat.codes.values
    dataset.UserID = dataset.UserID.astype('category').cat.codes.values
    dataset.MovieID = dataset.MovieID.astype('category').cat.codes.values
    dataset.to_pickle('./preprocessing2.pkl')
    
    return dataset

def TSNE_vis(model,emd_name,train_dataset,ItemInfo):
    movie_df = pd.read_csv(ItemInfo, sep="::")
    # Only obtain one genres
    movie_df["Genres"] = movie_df["Genres"].apply(lambda x:x.split("|")[0]) 
    movie_id = train_dataset.MovieID.unique()
    
    y = []
    for i in range(len(movie_id)):
        id = movie_id[i]
        genres = movie_df.loc[movie_df["movieID"]==id,"Genres"].tolist()
        y+=genres
    
    movie_emd = np.array(model.get_layer(emd_name).get_weights()).squeeze()
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, perplexity=80)
    vis_data = tsne.fit_transform(movie_emd)
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_c = le.fit_transform(y)
    plt.figure(figsize=(16,9))
    cm = plt.cm.get_cmap("tab20", 18)
    sc = plt.scatter(vis_data[:,0], vis_data[:,1], c=y_c, cmap=cm)
    plt.colorbar(ticks=range(18))
    plt.clim(-0.5, 17.5)
    plt.show()   
    pd.DataFrame({"class":list(range(18)),"genres":list(le.classes_)})

def main():
 
#    model_dnn = Building_new_model('DNN',n_users, n_movies,BIAS,N_LATENT_FACTORS_USER,N_LATENT_FACTORS_MOVIE)
    ###### Loading the model ######
#    from keras.models import load_model   
    N_LATENT_FACTORS = 128
    N_LATENT_FACTORS_USER = 32
    N_LATENT_FACTORS_MOVIE = 32
    BIAS = True
    NORMALIZE = 'mMfs'
    
    ###### Read raw dataset ######
    raw_dataset = pd.read_csv(RAW_PATH)
    ###### Split testing data from training data ######
    train_dataset, test_dataset = Train_test_split(raw_dataset) 
    n_users = len(train_dataset.UserID.unique())
    n_movies = len(train_dataset.MovieID.unique())
    ###### Normalize ######
#    train_norm, test_norm = train_dataset.copy(), test_dataset.copy()
#    train_norm.Rating = Do_normalize(train_dataset.Rating,NORMALIZE)
#    test_norm.Rating = Do_normalize(test_dataset.Rating,NORMALIZE)
    
    ###### Building the model ######
    model = Building_new_model(MODEL,n_users, n_movies,BIAS,N_LATENT_FACTORS)
#    model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})    
    
    t_start = time.time()
    ###### Training model ######
    history, model_name = Training_model(train_dataset, model)
    print('=== Spent %s seconds ===' % np.round((time.time() - t_start), 3))
    
    ###### Evaluate the model ######
    Evaluate_model(test_dataset, model)
  
    ###### Plot the history ######
#    history = np.load(HIS_PATH)
    Plot_history(history,model_name)
    
    ###### Plot the TSEN #######
#    TSNE_vis(model,'Movie-Embedding',train_dataset,ITEM_PATH)
    
    ###### Read observing dataset ######
    ob_dataset = pd.read_csv(OBSERVE_PATH)
    ###### Predicting the observing data ######
    Predict_and_submission(ob_dataset, SUBMIS_PATH, model)

if __name__ == '__main__':
#    args = parse_args()
#    main(args) 
    main()
