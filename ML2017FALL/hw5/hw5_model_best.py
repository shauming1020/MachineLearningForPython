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

MODEL = 'NNMF' # NNMF:Non-negative Matrix Factorisation  
MODEL_DIR = './model'
MODEL_NAME = MODEL+'-00090-0.45280.h5'
MODEL_PATH = MODEL_DIR + '/' + MODEL_NAME
HIS_DIR = './history'
HIS_PATH = HIS_DIR + '/NNMF-0.452804_history.npz'
PIC_DIR = './picture'
  
def Train_test_split(dataset):
    print('=== Split the dataset to train & test data ===')
    TEST_SIZE = 0.01
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

def Building_new_model(MODEL,n_users, n_movies,n_latent_factors=16,m_latent_factors=16):
    import keras
    from keras.optimizers import Adam
 
    #### Parameters ####
    WORD_LENGTH = 1 # an UserID is a value to see as a word for embedding layer.
    GENDER_LENGTH = 1
    Age_LENGTH = 1
    GENRES_LENGTH = 18
    OCCU_LENGTH = 21
    LMBDA = 0.00001
    OPT = 'adam'
    ####################
    
    if MODEL == 'NNMF':
        print('=== Non-negative Matrix Factorisation(NNMF) in keras ===')
        print('Build the NNMF model.')
        from keras.constraints import non_neg
        from keras.layers import Input, Embedding, Flatten, Dense, Dropout
        from keras.layers.merge import Dot, Add, Concatenate
        from keras.regularizers import l2
        from keras.models import Model
        # Inputs
        input_user = Input(shape=[WORD_LENGTH],name='User')
        input_movie = Input(shape=[WORD_LENGTH],name='Movie')
        input_gender = Input(shape=[GENDER_LENGTH],name='Gender')
        input_age = Input(shape=[Age_LENGTH],name='Age')
        input_occu = Input(shape=[OCCU_LENGTH],name='Occupation')
        input_genres = Input(shape=(GENRES_LENGTH,),name='Genres') 
        
        # Embeddings
        embedding_user = Embedding(n_users,
                                   n_latent_factors,
                                   name='NonNegUser-Embedding',
                                   embeddings_constraint=non_neg())(input_user)
        embedding_movie = Embedding(n_movies,
                                    n_latent_factors,
                                    name='NonNegMovie-Embedding',
                                    embeddings_constraint=non_neg())(input_movie)
        vec_user = Flatten(name='FlattenUsers')(embedding_user)   
        vec_movie = Flatten(name='FlattemMovies')(embedding_movie)     
        vec_occu = Dense(n_latent_factors, activation='linear')(input_occu)
        vec_gernes = Dense(n_latent_factors, activation='linear')(input_genres)
        # Dropout
        drop_user = Dropout(0.4, name='DropoutUsers')(vec_user)
        drop_movie = Dropout(0.4, name='DropoutMovies')(vec_movie)
        drop_occu = Dropout(0.4, name='DropoutOccu')(vec_occu)
        drop_gernes = Dropout(0.4, name='DropoutGernes')(vec_gernes)
        # Dot
        dot1 = Dot(axes=1,name='User-Movie')([drop_user, drop_movie])
        dot2 = Dot(axes=1,name='User-Occu')([drop_user,drop_occu])
        dot3 = Dot(axes=1,name='User-Gernes')([drop_user,drop_gernes])
        dot4 = Dot(axes=1,name='Movie-Occu')([drop_movie,drop_occu])
        dot5 = Dot(axes=1,name='Movie-Gernes')([drop_movie,drop_gernes])
        dot6 = Dot(axes=1,name='Occu-Gernes')([drop_occu,drop_gernes])
        
        # Concatenate
        con_dot = Concatenate()([dot1,dot2,dot3,dot4,dot5,dot6,input_gender,input_age])
        dense_out = Dense(1,activation='linear')(con_dot)
        # Bias
        embedding_user_bias = Embedding(n_users, 1,
                                        embeddings_initializer='zeros',
                                        name='User_bias-Embedding')(input_user)       
        embedding_movie_bias = Embedding(n_movies, 1,
                                         embeddings_initializer='zeros',
                                         name='Movie_bias-Embedding')(input_movie)
        vec_user_bias = keras.layers.Flatten(name='FlattenUsersBias')(embedding_user_bias) 
        vec_movie_bias = Flatten(name='FlattenMoviesBias')(embedding_movie_bias)
        # Output
        out = Add(name='Add')([vec_user_bias, vec_movie_bias, dense_out])            
        model = Model([input_user, input_movie, input_gender, input_age, input_occu, input_genres], out)
    
    print('model compile...')
    print('loss: mse, acc: binary, early_stopping loss: rmse') 
    model.compile(optimizer=OPT,loss= 'mean_absolute_error',metrics=['accuracy',rmse])
    model.summary()
    print('save the model...')
    model.save(MODEL_DIR+ '/' + MODEL+ '-00000-0.h5')
    return model

def Training_model(train_dataset, model, now):
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    ################ Parameters ################
    BATCH_SIZE = 10240
    EPOCHS = 1000
    VAL_SPLIT = 0.05
    PATIENCE = 30
    print('Number of UserID :',len(train_dataset.UserID.unique()), 
      ',Number of MovieID : ',len(train_dataset.MovieID.unique()))
    ############################################ 
    print('==================== Start Traning... ====================')
    # Shuffle training data
    train_dataset = train_dataset.sample(frac=1)
    Input_Occu = np.asarray(train_dataset["User_Occupation"].values.tolist())
    Input_Genres = np.asarray(train_dataset["Movie_Genres"].values.tolist())
    
    # 'rmse' as a monitor to check the current loss, instead of 'mse' .
    cp = ModelCheckpoint(MODEL_DIR+ '/' + MODEL + "_" + str(now) + "-{epoch:05d}-{val_rmse:.5f}.h5",
                     monitor='val_rmse', save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_rmse', patience=PATIENCE, mode='min')
    history = model.fit([train_dataset.UserID, train_dataset.MovieID, train_dataset.User_Gender,
                         train_dataset.User_Age,Input_Occu,Input_Genres],
                        train_dataset.Rating, 
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
    BATCH_SIZE = 10240
    Input_Occu = np.asarray(test_dataset["User_Occupation"].values.tolist())
    Input_Genres = np.asarray(test_dataset["Movie_Genres"].values.tolist())
    loss, acc, rmse = model.evaluate([test_dataset.UserID, test_dataset.MovieID, test_dataset.User_Gender,
                                      test_dataset.User_Age,Input_Occu,Input_Genres], 
                                      test_dataset.Rating,batch_size=BATCH_SIZE)
    print('loss: %.5f'%loss, 'acc: %.5f'%acc, 'rmse: %.5f'%rmse)
    return np.round(loss,5), np.round(acc,5), np.round(rmse,5)
 
def Predict_and_submission(ob_dataset, submission, model):
    print('=== Predicting ... ===')
    Input_Occu = np.asarray(ob_dataset["User_Occupation"].values.tolist())
    Input_Genres = np.asarray(ob_dataset["Movie_Genres"].values.tolist())
    pred_rating = model.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                 ob_dataset.User_Gender,ob_dataset.User_Age,
                                 Input_Occu,Input_Genres])
    pred_rating = np.clip(pred_rating,1,5) 
    print('=== Output the sample submission===')
    sampleSubmission = pd.read_csv(submission)
    sampleSubmission["Rating"] = pred_rating  
    sampleSubmission.to_csv(submission,index=None)    
    return pred_rating

def Preprocess_data(dataset,UserInfo,ItemInfo,SAVE_NAME):
    user_info = pd.read_csv(UserInfo, sep="::")
    user_id = dataset.UserID
    movie_info = pd.read_csv(ItemInfo, sep="::")
    movie_info["Genres"] = movie_info["Genres"].str.split('|')
    movie_id = dataset.MovieID
    y = []        
    df = pd.DataFrame(columns=['Gender','Age','Occupation'])
    for i in range(len(dataset)):
        print("now deal with i/total ",i,"/",len(dataset))
        id = user_id[i]
        info = user_info.loc[user_info["UserID"]==id, ("Gender","Age","Occupation")]
        df = pd.concat([df,info],ignore_index=True)
        id = movie_id[i]
        genres = movie_info.loc[movie_info["movieID"]==id,"Genres"].tolist()
        y+=genres
    dataset['User_Gender'] = df.Gender.astype(bool).astype(int)
    dataset['User_Age'] = df.Age
    dataset['User_Occupation'] = df.Occupation
    dataset['Movie_Genres'] = y

    from sklearn.preprocessing import OneHotEncoder
    movie_info = pd.read_csv(ItemInfo, sep="::")
    movie_id_uni = movie_info["Genres"].apply(lambda x:x.split("|")[0])
    movie_id_uni = movie_id_uni.unique().reshape(-1,1)
    enc_m = OneHotEncoder()
    enc_m.fit(movie_id_uni)
    y = []
    for i in range(len(dataset['Movie_Genres'])):
        print("now encode i/total ",i,"/",len(dataset['Movie_Genres']))
        ec = dataset['Movie_Genres'][i]
        ec = np.asarray(ec).reshape(-1,1)
        onehot = enc_m.transform(ec).toarray().sum(axis=0)
        y.append(onehot)
    dataset['Movie_Genres'] = y   
    
    enc_u = OneHotEncoder()
    occ = dataset["User_Occupation"].values
    occ = enc_u.fit_transform(occ.reshape(-1,1)).toarray()
    dataset['User_Occupation'] = occ.astype(int).tolist()
    
    dataset.to_pickle('./'+ SAVE_NAME +'.pkl')
    
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
    N_LATENT_FACTORS = 128
    N_LATENT_FACTORS_USER = 32
    N_LATENT_FACTORS_MOVIE = 32
    NORMALIZE = 'mMfs'
    
    ###### Read raw dataset ######
#    raw_dataset = pd.read_csv(RAW_PATH)
 
    ###### Preprocessing ######
#    raw_dataset = Preprocess_data(raw_dataset,USER_PATH,ITEM_PATH,'preprocessing')
    import pickle
    with open('preprocessing.pkl', 'rb') as file:
        raw_dataset =pickle.load(file)
    
    ###### Split testing data from training data ######
    train_dataset, test_dataset = Train_test_split(raw_dataset) 
    n_users = len(train_dataset.UserID.unique())
    n_movies = len(train_dataset.MovieID.unique())
    
    
    ###### Normalize ######
#    train_norm, test_norm = train_dataset.copy(), test_dataset.copy()
#    train_norm.Rating = Do_normalize(train_dataset.Rating,NORMALIZE)
#    test_norm.Rating = Do_normalize(test_dataset.Rating,NORMALIZE)
    
    ###### Building the model ######
    model = Building_new_model(MODEL,n_users, n_movies,N_LATENT_FACTORS)
#    model_dnn = Building_new_model('DNN',n_users, n_movies,BIAS,N_LATENT_FACTORS_USER,N_LATENT_FACTORS_MOVIE)
    ###### Loading the model ######
#    from keras.models import load_model
#    model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})   
    
#    from IPython.display import SVG
#    from keras.utils.vis_utils import model_to_dot   
#    SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
#    
    t_start = time.time()
    ###### Training model ######
    history, model_name = Training_model(train_dataset, model)
    print('=== Spent %s seconds ===' % np.round((time.time() - t_start), 3))
    
    ###### Evaluate the model ######
    Evaluate_model(test_dataset, model)
  
    ###### Plot the history ######
#    history = np.load(HIS_PATH)
    Plot_history(history,model_name)
    
#    ###### Plot the TSEN #######
#    TSNE_vis(model,'NonNegMovie-Embedding',train_dataset,ITEM_PATH)
    
    ###### Read observing dataset ######
#    ob_dataset = pd.read_csv(OBSERVE_PATH)
#    ob_dataset = Preprocess_data(ob_dataset,USER_PATH,ITEM_PATH,'ob_preprocessing')
    
    import pickle
    with open('ob_preprocessing.pkl', 'rb') as file:
        ob_dataset = pickle.load(file)
    ###### Predicting the observing data ######
    Predict_and_submission(ob_dataset, SUBMIS_PATH, model)

def ensemble():
    
    N_LATENT_FACTORS = 128
    
    import pickle
    with open('preprocessing.pkl', 'rb') as file:
        train_dataset = pickle.load(file)    
    
    import pickle
    with open('ob_preprocessing.pkl', 'rb') as file:
        ob_dataset = pickle.load(file)
    
    train_dataset['User_Age'] = Do_normalize(train_dataset['User_Age'],'std')
    ob_dataset['User_Age'] = Do_normalize(ob_dataset['User_Age'],'std')
    
    import keras
    for i in range(1,10):      
        model = Building_new_model(MODEL,6040,3688,N_LATENT_FACTORS)  
        history, model_name = Training_model(train_dataset, model, i)
        
    pred = []
    from keras.models import load_model
    model_1 = load_model('model/NNMF_1-00021-0.91895.h5', custom_objects={'rmse': rmse})
    model_2 = load_model('model/NNMF_2-00196-0.86360.h5', custom_objects={'rmse': rmse})
    model_3 = load_model('model/NNMF_3-00117-0.86873.h5', custom_objects={'rmse': rmse})
    model_4 = load_model('model/NNMF_4-00150-0.86664.h5', custom_objects={'rmse': rmse})
    model_5 = load_model('model/NNMF_5-00129-0.87071.h5', custom_objects={'rmse': rmse})
    model_6 = load_model('model/NNMF_6-00288-0.86107.h5', custom_objects={'rmse': rmse})
    model_7 = load_model('model/NNMF_7-00257-0.86533.h5', custom_objects={'rmse': rmse})
    model_8 = load_model('model/NNMF_8-00114-0.86025.h5', custom_objects={'rmse': rmse})
    model_9 = load_model('model/NNMF_9-00110-0.87533.h5', custom_objects={'rmse': rmse})
    model_10 = load_model('model/NNMF_10-00071-0.87601.h5', custom_objects={'rmse': rmse})
    model_11 = load_model('model/NNMF_11-00036-0.90762.h5', custom_objects={'rmse': rmse})
    model_12 = load_model('model/NNMF_12-00038-0.91009.h5', custom_objects={'rmse': rmse})
    model_13 = load_model('model/NNMF_13-00107-0.87914.h5', custom_objects={'rmse': rmse})     
    model_14 = load_model('model/NNMF_14-00161-0.86762.h5', custom_objects={'rmse': rmse})  
       
    Input_Occu = np.asarray(ob_dataset["User_Occupation"].values.tolist())
    Input_Genres = np.asarray(ob_dataset["Movie_Genres"].values.tolist())
    pred1 = model_1.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred2 = model_2.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred3 = model_3.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred4 = model_4.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred5 = model_5.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred6 = model_6.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred7 = model_7.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred8 = model_8.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred9 = model_9.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred10 = model_10.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred11 = model_11.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred12 = model_12.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred13 = model_13.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    pred14 = model_14.predict([ob_dataset.UserID, ob_dataset.MovieID,
                                     ob_dataset.User_Gender,ob_dataset.User_Age,
                                     Input_Occu,Input_Genres])
    
    pred = np.average(np.concatenate((pred1,pred2,pred3,pred4,pred5,pred6,pred7,
                                      pred8,pred9,pred10,pred11,pred12,pred13,pred14),
                        axis=-1),axis=-1)
    pred = np.clip(pred,1,5) 
    sampleSubmission = pd.read_csv(SUBMIS_PATH)
    sampleSubmission["Rating"] = pred  
    sampleSubmission.to_csv(SUBMIS_PATH,index=None)    
    

if __name__ == '__main__':
#    args = parse_args()
#    main(args) 
    
    ensemble()
