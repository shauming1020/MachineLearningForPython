# HW1-Linear Regression
import sys
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

### Parameters ###
RAWDATA_PATH = './dataset/train.csv'
OBSERVE_PATH = './dataset/test.csv'
SUBMISSION_PATH = './dataset/sampleSubmission.csv'
#RAWDATA_PATH = sys.argv[1]
#OBSERVE_PATH = sys.argv[2]
#SUBMISSION_PATH = sys.argv[3]
MODE_PATH = './model'
HIS_PATH = './history'
PIC_PATH = './picture'

FEATURES = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5',
            'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
##################

def Create_dict(raw, select='raw'):
    where_are_NaNs = np.isnan(raw)
    raw[where_are_NaNs] = 0
    month_to_data = {}
    if select == 'raw':
        raw = raw[1:,3:]
        for month in range(12):
            sample = np.empty(shape = (18 , 480))
            for day in range(20):
                for hour in range(24): 
                    sample[:,day * 24 + hour] = raw[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
            month_to_data[month] = sample
    elif select == 'ob':  
        raw = raw[:, 2: ]  
        for id in range(240):
            month_to_data[id] = raw[18 * id : 18 * (id+1),:]
    return month_to_data


def Preprocess(month_to_data, features, f_size, predict_term, select='raw'):
    n_features = len(features)
    id_features = []
    id_predict = []
    # Get id in features
    for i, name in enumerate(FEATURES):
        if name == predict_term:
            id_predict.append(i)
        for f in features:
            if f == name:
                id_features.append(i)               
    # Modify the dict, the last row is wanted to predict vaules.
    month_to_datay = {}
    for month, data in month_to_data.items():
        month_to_data[month] = data[id_features]
        month_to_datay[month] = data[id_predict]        
    ## Select the model to preprocess the data
    if select == 'raw':
        # Sample every 10 hrs: previous 9-hr as train-feature, 10th-hr as train-predict
        # Every month's 0~9 hrs cannot be predict.
        x = np.empty(shape = (12 * (480-f_size) , n_features * f_size),dtype = float)
        y = np.empty(shape = (12 * (480-f_size) , 1),dtype = float)
        for month in range(12): 
            for day in range(20): 
                for hour in range(24):   
                    if day == 19 and hour > (23-f_size):
                        continue  
                    x[month * (480-f_size) + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + f_size].reshape(1,-1) 
                    y[month * (480-f_size) + day * 24 + hour,0] = month_to_datay[month][:,day * 24 + hour + f_size] # Predict term
        return x,y
    elif select == 'ob':
        if f_size == 9:
            x = np.empty(shape = (240, 18 * 9),dtype = float)
            y = np.empty(shape = (240, 1),dtype = float)
            for idx in range(240):
                x[idx,:] = month_to_data[idx][:,:].reshape(1,-1) 
            x_p, y_p = x,y
        elif f_size < 9:
            x = np.empty(shape = (240 * (9-f_size), n_features * f_size),dtype = float)
            y = np.empty(shape = (240 * (9-f_size), 1),dtype = float)  
            x_p = np.empty(shape = (240, n_features * f_size),dtype=float)
            y_p = np.empty(shape = (240, 1),dtype=float)    
            # get the train data from ob data
            for idx in range(240):
                for hour in range(9):
                    # get the feature to predict
                    if hour == 9-f_size:
                        x_p[idx,:] = month_to_data[idx][:,hour : hour + f_size].reshape(1,-1)
                    if hour > 8-f_size:
                        continue
                    x[idx * (9-f_size) + hour,:] = month_to_data[idx][:,hour : hour + f_size].reshape(1,-1)
                    y[idx * (9-f_size) + hour,0] = month_to_datay[idx][:,hour + f_size]
        return x,y,x_p,y_p

def Normalization(x,mean='None',std='None',trans=False):
    ## Normalization
    if trans == False:
        mean = np.mean(x, axis = 0) 
        std = np.std(x, axis = 0)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]
    if trans:
        return x
    else:
        return x,mean,std

def Split_train_test(x,y,train_rate=0.7):
    perm = np.random.permutation(len(y))
    x, y = x[perm], y[perm]
    split_pos = np.round(len(y)*train_rate).astype('int32')
    return x[:split_pos], x[split_pos:], y[:split_pos], y[split_pos:]

def Fit(X,y,val=None,validation_split=0.0,bias=True,weights='zeros',
      loss='rmse',opt='vanilla',regularization=None,lamda=0.0005,learning_rate=200,
      epochs=1,batch_size=None,patience=None,save_model=True):

    def initial(X,weights,bias):    
        dim = X.shape[1]
        if bias:
            dim = dim + 1
            X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1).astype(float)
        if weights == 'zeros':
            w = np.zeros(shape=(dim, 1))
        elif weights == 'random':
            w = np.random.rand(dim, 1)
        return X, w
    # loss_value      
    def loss_value(X,y,w,loss):
        if loss == 'mse':
            pred = X.dot(w)
            square_error = (y-pred)**2
            mse = square_error.sum()/len(y)
            return mse
        elif loss == 'rmse':
            mse = loss_value(X,y,w,'mse')
            return np.power(mse,0.5)
    # Optmize function
    def update_parameters(X,y,w,sigma,opt):
        if regularization == 'l1':
            reg = lamda * w
        elif regularization == 'l2':
            reg = 2. * lamda * w
        elif regularization == None:
            reg = lamda
        pred = X.dot(w)
        grad = -2. * np.dot(X.T,y-pred) + reg
        if opt == 'ada':
            sigma += np.sqrt(grad**2)
            w -= grad * learning_rate / sigma 
        else:
            w -= grad * learning_rate
        return w, sigma
    
    def get_mini_batch(X,y,batch_size):
        mini_batches = []
        data = np.hstack((X,y))
        np.random.shuffle(data) 
        n_minibatches = data.shape[0] 
        for i in range(n_minibatches + 1): 
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        if data.shape[0] % batch_size != 0: 
            mini_batch = data[i * batch_size:data.shape[0]] 
            X_mini = mini_batch[:, :-1] 
            Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
            mini_batches.append((X_mini, Y_mini)) 
        return mini_batches

    # Split Training and Validation Set
    if val:
        X_t,X_v,y_t,y_v = Split_train_test(X,y,(1-validation_split))
    else:
        X_t,X_v,y_t,y_v = X,X,y,y
    
    # Fitting summary
    print('training sample:',len(X_t),' validation sample:',len(X_v))

    # Initialization
    X_t, w = initial(X_t,weights,bias)
    X_v, w_v = initial(X_v,weights,bias)  
    # Set model parameter
    sigma = None
    if opt == 'stg':
        batch_size = 1
    elif opt == 'ada':
        sigma = np.zeros(w.shape)   
    # Mini-Batch Training    
    cost_history = []
    see_best_cost,see = None,0
    best_w = np.copy(w) # if we use '=', it will assign the same address.
    for epoch in range(epochs):
        # Shuffle 
        perm = np.random.permutation(len(y_t))
        X_t, y_t = X_t[perm], y_t[perm]
        
        if batch_size == None:
                # Update weights
                w, sigma = update_parameters(X_t,y_t,w,sigma,opt)    
        # Using mini-batch, get batch data
        else:
            mini_batches = get_mini_batch(X_t, y_t, batch_size)
            for mini_batch in mini_batches: 
                X_mini, y_mini = mini_batch 
                # Update weights
                w, sigma = update_parameters(X_mini,y_mini,w,sigma,opt)
        # Cost
        cost = loss_value(X_t,y_t,w,loss) # on training set
        cost_v = loss_value(X_v,y_v,w,loss) # on validation ser
        cost_history.append((cost,cost_v))       
        print('Epoch:'+str(epoch) + '/' + str(epochs),'-loss: %.3f' % cost," val loss: %.3f" % cost_v)
        # Early Stopping - Avoid Overfitting
        if patience != None:
            see += 1
            if see_best_cost == None:
                see_best_cost = np.round(cost,decimals=3)    
            elif see_best_cost > cost:   
                see_best_cost = np.round(cost,decimals=3)
                see = 0
                best_w = np.copy(w)
                continue
            elif see_best_cost < cost:      
                print('The current loss %.3f'%cost,'is not best than',see_best_cost) 
                if see < patience:
                    continue
                else:
                    print('early stopping',see,'/',patience)
                    break
    # Save the best model
    if save_model != False:
        if patience != None:
            np.save(MODE_PATH+'/'+save_model+'_weight.npy',best_w) # save best weight
        else:
            np.save(MODE_PATH+'/'+save_model+'_weight.npy',w)
        np.save(HIS_PATH+'/'+save_model+'_history.npy',np.asarray(cost_history)) # save history
    return w, np.asarray(cost_history)

def Plot_loss_history(history,save_model):
    plt.clf()
    loss_t, loss_v = history[:,0], history[:,1]
    plt.plot(loss_t,'b')
    plt.plot(loss_v,'r')
    plt.legend(['loss', 'val_loss'], loc="upper left")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Process")
    plt.show()
    if save_model == False:     
        plt.savefig(PIC_PATH+'/_history.png')
    else:
        plt.savefig(PIC_PATH+'/'+save_model+'_history.png')
    plt.close()

def Sample():
    ## Read in raw data
    with open(RAWDATA_PATH) as f:
        raw = np.genfromtxt(f,delimiter=',')
    ## Create the data dictory
    month_to_data = Create_dict(raw)
    ## Preprocess to split features and true data
    features = FEATURES
    f_size = 9
    predict_term = 'PM2.5'
    X, y = Preprocess(month_to_data, features, f_size, predict_term)
    ## Normalization
    X,mean,std = Normalization(X)
    ## Split to training and testing set
#    X_train, X_test, y_train, y_test = Split_train_test(X,y,train_rate=0.9)
    X_train, y_train = X,y
    
    ##  Training
    val=True
    validation_split=0.1
    bias=True
    weights='zeros'
    loss='rmse'
    opt='ada'
    regularization='l2'
    lamda= 0
    learning_rate=200
    epochs=1000
    batch_size=None
    patience=None
    save_model='sample'
    
    model, history = Fit(X_train,y_train,val,validation_split,bias,weights,
                      loss,opt,regularization,lamda,learning_rate,
                      epochs,batch_size,patience,save_model)
    
    ## Read in Observe set
    model = np.load(MODE_PATH+'/'+save_model+'_weight.npy')                               ## load weight
    ob_raw_data = np.genfromtxt(OBSERVE_PATH, delimiter=',')
    ob_to_data = Create_dict(ob_raw_data,'ob')
    ob_X,ob_y,ob_X_p,ob_y_p= Preprocess(ob_to_data, features, f_size, predict_term,'ob') 
    ob_X = Normalization(ob_X,mean,std,trans=True)

    # Add the bias-term to ob_X_p
    ob_X_p = np.concatenate((np.ones(shape = (ob_X_p.shape[0],1)),ob_X_p),axis = 1).astype(float)
    ob_y_p = ob_X_p.dot(model)  
    
    # Plot the history
    Plot_loss_history(history,save_model)
    
    ## Write file
    with open(SUBMISSION_PATH,"w") as f:
        w = csv.writer(f)
        title = ['id','value']
        w.writerow(title) 
        for i in range(240):
            content = ['id_'+str(i),ob_y_p[i][0]]
            w.writerow(content) 

def Best():
    ## Read in raw data
    with open(RAWDATA_PATH) as f:
        raw = np.genfromtxt(f,delimiter=',')
    ob_raw_data = np.genfromtxt(OBSERVE_PATH, delimiter=',')
    ## Create the data dictory
    month_to_data = Create_dict(raw)
    ob_to_data = Create_dict(ob_raw_data,'ob')
    ## Preprocess to split features and true data
    features = ['PM2.5','PM2.5','PM2.5']
    f_size = 8
    predict_term = 'PM2.5'
    X, y = Preprocess(month_to_data, features, f_size, predict_term)
    ob_X,ob_y,ob_X_p,ob_y_p= Preprocess(ob_to_data, features, f_size, predict_term,'ob') 
    
    ## Normalization
    X,mean,std = Normalization(np.vstack((X,ob_X)))
    ## Split to training and testing set
#    X_train, X_test, y_train, y_test = Split_train_test(X,y,train_rate=0.9)
    X_train, y_train = X, np.vstack((y,ob_y))
    
    ##  Training
    val=True
    validation_split=0.01
    bias=True
    weights='zeros'
    loss='rmse'
    opt='ada'
    regularization='l1'
    lamda= 0.01
    learning_rate=256
    epochs=1024
    batch_size=1024
    patience=None
    save_model='best'
    
    model, history = Fit(X_train,y_train,val,validation_split,bias,weights,
                      loss,opt,regularization,lamda,learning_rate,
                      epochs,batch_size,patience,save_model)

    ## Read in Observe set
    model = np.load(MODE_PATH+'/'+save_model+'_weight.npy')
    ob_X_p = Normalization(ob_X_p,mean,std,trans=True)
    # Add the bias-term to ob_X_p
    ob_X_p = np.concatenate((np.ones(shape = (ob_X_p.shape[0],1)),ob_X_p),axis = 1).astype(float)
    ob_y_p = ob_X_p.dot(model)  
    # Plot the history
    Plot_loss_history(history,save_model)
    ## Write file
    print('Submission...')
    with open(SUBMISSION_PATH,"w") as f:
        w = csv.writer(f)
        title = ['id','value']
        w.writerow(title) 
        for i in range(240):
            content = ['id_'+str(i),ob_y_p[i][0]]
            w.writerow(content) 

if __name__ == '__main__':    
    Best()