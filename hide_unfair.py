
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model
from keras import regularizers
from tensorflow.python.keras.utils.generic_utils import class_and_config_for_serialized_keras_object
from tqdm import tqdm
import numpy as np
import shap as shap_util
from itertools import combinations
from attribution import shapley_regression
import math
import tensorflow as tf
import keras
import copy

from keras import backend as K

import random
import os

def build_dataset(random_seed):
    X_raw,y_raw = shap_util.datasets.adult()
    X_display,y_display = shap_util.datasets.adult(display=True)
    dtypes = list(zip(X_raw.dtypes.index, map(str, X_raw.dtypes)))
    # normalize data (this is important for model convergence)
    for k,dtype in dtypes:
        X_raw[k] -= X_raw[k].mean()
        X_raw[k] /= X_raw[k].std()
    X_train, X_valid, y_train, y_valid = train_test_split(X_raw, y_raw, test_size=0.2, random_state=random_seed-3)
    X_train_array = np.array([X_train[k].values for k,t in dtypes]).T
    X_valid_array = np.array([X_valid[k].values for k,t in dtypes]).T
    return X_train, X_valid, y_train, y_valid, X_train_array, X_valid_array

### build a simple 4-layer regression model
def build_regression(X_train_array, X_valid_array, y_train, y_valid, random_seed): 
    input_els = Input(shape=(12,))
    layer1 = (Dense(100, activation="relu")(input_els))
    layer2 = Dropout(0.5)(Dense(100, activation="relu")(layer1))
    layer3 = Dropout(0.5)(Dense(100, activation="relu")(layer2))
    layer4 = Dropout(0.5)(Dense(100, activation="relu")(layer3))
    out = Dense(1, activation='sigmoid')(layer4)
    # train model
    regression = Model(inputs=input_els, outputs=[out])
    regression.compile(optimizer="adam", loss='mse')

    trained = True

    # train model
    if not trained:
        regression.fit(
            X_train_array,
            y_train+ X_train_array[:,7]*0.1,
            epochs=20,
            batch_size=512,
            shuffle=True,
            validation_data=(X_valid_array, y_valid + X_valid_array[:,7]*0.1)
        )
        regression.save_weights('./original_model_deep_{}.ckpt'.format(random_seed))
    else:
        regression.load_weights('./original_model_deep_{}.ckpt'.format(random_seed))
    return regression


def build_prob_data(regression, X_train_array, y_valid):
    sh1 = X_train_array.shape[0]
    X_train_array_fake = np.zeros((X_train_array.shape[0]*100, X_train_array.shape[1]))
    y_train_fake = np.zeros((X_train_array.shape[0]*100,3))
    # y_fake has size of (N, 3)
    # first dimension is prediction of regression
    # second dimension is whether on-manifold or off-manifold, 1 is on-manifold
    # third dimension is set to 0 now
    y_pred = regression.predict(X_train_array)[:,0]

    for i in range(50):
        X_train_array_fake[sh1*i: sh1*(i+1), :] = X_train_array
        y_train_fake[sh1*i: sh1*(i+1),0] = y_pred
        y_train_fake[sh1*i: sh1*(i+1),1] = 0

    for i in range(50):
        X_train_array_fake[(50+i)*sh1:(51+i)*sh1 , :] = X_train_array
        y_train_fake[(50+i)*sh1:(51+i)*sh1 ,1] = 1

    mask = np.reshape(np.random.choice(2,size=(X_train_array.shape[0])*50*12),((X_train_array.shape[0])*50,12))
    mask = np.concatenate([mask,np.zeros((X_train_array.shape[0]*50,X_train_array.shape[1]))],axis=0)

    X_train_array_fake[np.where(mask>0)] = 0
    for i in range(100):
        # X_train_array_fake[sh1*i: sh1*(i+1), :] = X_train_array
        y_train_fake[sh1*i: sh1*(i+1),0] = regression.predict(X_train_array_fake[sh1*i: sh1*(i+1), :])[:,0]

    return X_train_array_fake, y_train_fake

def build_prob_model(X_train_array_fake, y_train_fake, random_seed):
    # build prob_model (ooD detector), which predicts whether x is 1  (on-mainfold) 0 (off-manifold)
    input_prob = Input(shape=(12,))
    layer1_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(input_prob)
    layer2_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer1_prob)
    layer3_prob = Dense(200, activation="relu",kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer2_prob)
    out_prob = Dense(1)(layer3_prob)

    trained = True
    model_prob = Model(inputs=input_prob, outputs=[out_prob])
    opt = keras.optimizers.SGD(learning_rate=1e-3, momentum = 0.9)
    #model_prob.compile(optimizer=opt, loss='mean_squared_error')
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
    model_prob.compile(optimizer=opt, loss=loss)
    # train model
    if not trained:
        model_prob.fit(
            X_train_array_fake,
            y_train_fake[:,1],
            epochs=30,
            batch_size=1000,
            shuffle=True,
        )
        model_prob.save_weights('./prob_model_deep_{}.ckpt'.format(random_seed))
    else:
        model_prob.load_weights('./prob_model_deep_{}.ckpt'.format(random_seed))
    return model_prob

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# transfer ODD prob to density, 1/17 is an arbitrary constant
def prob_NCE(x):
    h = model_prob.predict(x)
    G =  np.clip(sigmoid(h),0.01,0.99)
    return G / (1-G) / 17

def build_regression_fake_data(regression, X_train_array_fake, y_train_fake, X_valid_array):
    # third dimension of y_train_fake is set to 1- prob_NCE, and values less than 0.8 are clipped to 0
    # the idea is to find how OOD is the particular x, if it is OOD, prob_NCE will be small, and this dim should be large
    y_valid_fake = np.zeros((y_valid.shape[0],3))
    sh1 = X_train_array.shape[0]
    for i in range(50):
        y_train_fake[sh1*i: sh1*(i+1),2:] = 1- prob_NCE(X_train_array_fake[sh1*i: sh1*(i+1), :])

    y_train_fake[np.where(X_train_array_fake[:,7]<=0),2] = 0
    y_train_fake[np.where(y_train_fake[:,2]<0.8),2] = 0.0
    
    # X_train_array_fake2 just set the feature sex to 0
    X_train_array_fake2 = copy.copy(X_train_array_fake)
    X_train_array_fake2[:(X_train_array.shape[0])*50,7] = 0
    y_valid_fake[:,0:1] = regression.predict(X_valid_array)
    y_valid_fake[:,1] = 1
    kk = regression.predict(X_train_array_fake)
    y_train_fake[:,0:1] = kk
    
    return X_train_array_fake, X_train_array_fake2, y_train_fake, y_valid_fake

def build_regression_fake(X_train_array_fake, X_train_array_fake2, y_train_fake, y_valid_fake, random_seed):
    input_els1 = Input(shape=(12,))
    input_els2 = Input(shape=(12,))
    fc1 = Dense(1000, activation="relu")
    fc2 = Dense(1000, activation="relu")
    fc3 = Dense(1000, activation="relu")
    fc4 = Dense(1000, activation="relu")
    #fc2= Dense(1000, activation="relu")
    out1 = Dense(1)

    fc11 = Dropout(0.5)(fc4(fc3(fc2(fc1(input_els1)))))
    #fc21 = Dropout(0.5)(fc2(fc11))
    out11 = out1(fc11)

    fc12 = Dropout(0.5)(fc4(fc3(fc2(fc1(input_els2)))))
    #fc22 = Dropout(0.5)(fc2(fc12))
    out12 = out1(fc12)

    out_final = Concatenate(axis=1)([out11, out12])


    def fake_regression_loss(y_true, y_pred):
        y_pred1 = y_pred[:,0:1]
        y_pred2 = y_pred[:,1:]
        y_true1 = y_true[:,0:1]
        y_true2 = y_true[:,1:2]
        y_true3 = y_true[:,2:]
        y_true1_float = tf.cast(y_true1,'float32')
        y_true2_float = tf.cast(y_true2,'float32')
        y_true3_float = tf.cast(y_true3,'float32')
        # this is squared error between prediction of fake model and true model, only applied on on-manifold regions
        square_error = tf.reduce_mean(tf.math.pow((y_pred1-y_true1_float)*y_true2_float, 2))
        # this is the squared error between prediction of fake model and (prediction of fake model with sex feature set to 0 -0.2)
        # this is only applied on off-manifold regions
        square_error2 = tf.reduce_mean(tf.math.pow(y_pred1-(y_pred2-0.2), 2)*y_true3_float)
        # with these two loss, the fake regression will behave like normal regression on on-manifold regions
        # but have negatively with repect to the sex feature on off-manifold regions.

        return square_error + 30.0*square_error2

    # train new fake model
    regression_fake = Model(inputs=[input_els1, input_els2], outputs=out_final)
    regression_fake.compile(optimizer="adam", loss=fake_regression_loss)
    trained = False
    if not trained:
        regression_fake.fit(
        [X_train_array_fake,X_train_array_fake2],
        y_train_fake,
        epochs=10,
        batch_size=512,
        shuffle=True,
        validation_data=([X_valid_array,X_valid_array], y_valid_fake)
        )
        regression_fake.save_weights('./regression_fake_deep_{}.ckpt'.format(random_seed))
    else:
        regression_fake.load_weights('./regression_fake_deep_{}.ckpt'.format(random_seed))
    return regression_fake

def build_CES_SUP_data(regression, model_prob, X_train_array, X_valid_array, y_valid):
    sh1 = X_train_array.shape[0]
    X_train_array_fake = np.zeros((X_train_array.shape[0]*100, X_train_array.shape[1]))
    y_train_fake = np.zeros((X_train_array.shape[0]*100,3))
    y_valid_fake = np.zeros((y_valid.shape[0],3))
    y_valid_fake[:,0:1] = regression.predict(X_valid_array)
    y_valid_fake[:,1] = 0
    y_pred = regression.predict(X_train_array)[:,0]

    for i in range(50):
        X_train_array_fake[sh1*i: sh1*(i+1), :] = X_train_array
        y_train_fake[sh1*i: sh1*(i+1),0] = y_pred
        y_train_fake[sh1*i: sh1*(i+1),1] = 0

    for i in range(50):
        X_train_array_fake[(50+i)*sh1:(51+i)*sh1 , :] = X_train_array
        y_train_fake[(50+i)*sh1:(51+i)*sh1 :,0] = y_pred
        y_train_fake[(50+i)*sh1:(51+i)*sh1 ,1] = 1
        y_train_fake[(50+i)*sh1:(51+i)*sh1 ,2] = 0

    mask = np.reshape(np.random.choice(2,size=(X_train_array.shape[0])*50*12),((X_train_array.shape[0])*50,12))
    mask = np.concatenate([mask,np.zeros((X_train_array.shape[0]*50,X_train_array.shape[1]))],axis=0)

    X_train_array_fake[np.where(mask>0)] = 0
    X_train_array_fake2 = copy.copy(X_train_array_fake)
    X_train_array_fake2[:(X_train_array.shape[0])*50,7] = 0

    for i in range(50):
        y_train_fake[sh1*i: sh1*(i+1),2:] = 1- prob_NCE(X_train_array_fake[sh1*i: sh1*(i+1), :])

    y_train_fake[np.where(X_train_array_fake[:,7]<=0),2] = 0
    y_train_fake[np.where(y_train_fake[:,2]<0.99),2] = 0.0


    mask = np.reshape(np.random.choice(2,size=(X_valid_array.shape[0])*12),((X_valid_array.shape[0]),12))
    X_valid_array_fake = copy.copy(X_valid_array)
    X_valid_array_fake[np.where(mask>0)] = 0

    return X_train_array_fake, X_train_array_fake2, X_valid_array_fake, y_train_fake, y_valid_fake
    
def build_CES_SUP(X_train_array_fake, X_train_array_fake2, X_valid_array_fake, y_train_fake, y_valid_fake, random_seed):
    def custom_loss_ces1(y_true, y_pred):
        y_pred1 = y_pred[:,0:1]
        y_pred2 = y_pred[:,1:]
        y_true1 = y_true[:,0:1]
        y_true2 = y_true[:,1:2]
        y_true3 = y_true[:,2:]
        y_true1_float = tf.cast(y_true1,'float32')
        y_true2_float = tf.cast(y_true2,'float32')
        y_true3_float = tf.cast(y_true3,'float32')
        print(y_true.shape)
        print(y_pred.shape)
        #cross_entropy = tf.reduce_mean(keras.losses.binary_crossentropy(y_true1, y_pred1, from_logits=True))
        #return cross_entropy + 0.3*tf.reduce_mean(tf.math.pow(y_pred1-y_pred2, 2))

        # this is squared error between prediction of fake model and true model, only applied on off-manifold regions
        square_error = tf.reduce_mean(tf.math.pow((y_pred1-y_true1_float)*(1-y_true2_float), 2))
        return square_error 



    def custom_loss_ces2(y_true, y_pred):
        y_pred1 = y_pred[:,0:1]
        y_pred2 = y_pred[:,1:]
        y_true1 = y_true[:,0:1]
        y_true2 = y_true[:,1:2]
        y_true3 = y_true[:,2:]
        y_true1_float = tf.cast(y_true1,'float32')
        y_true2_float = tf.cast(y_true2,'float32')
        y_true3_float = tf.cast(y_true3,'float32')
        print(y_true.shape)
        print(y_pred.shape)
        #cross_entropy = tf.reduce_mean(keras.losses.binary_crossentropy(y_true1, y_pred1, from_logits=True))
        #return cross_entropy + 0.3*tf.reduce_mean(tf.math.pow(y_pred1-y_pred2, 2))

        # this is squared error between prediction of fake model and true model, only applied on off-manifold regions
        square_error = tf.reduce_mean(tf.math.pow((y_pred1-y_true1_float)*(1-y_true2_float), 2))
        # this is the squared error between prediction of fake model and (prediction of fake model with sex feature set to 0 - 0.05)
        # this is only applied on off-manifold regions
        square_error2 = tf.reduce_mean(tf.math.pow(y_pred1-(y_pred2-0.05), 2)*y_true3_float)

        return square_error + 0.3*square_error2


    # train CES sup on orig model 
    input_els1 = Input(shape=(12,))
    input_els2 = Input(shape=(12,))

    fc1 = Dense(1000, activation="relu")
    fc2 = Dense(1000, activation="relu")
    fc3 = Dense(1000, activation="relu")
    fc4 = Dense(1000, activation="relu")
    #fc2= Dense(1000, activation="relu")
    out1 = Dense(1, activation='sigmoid')

    fc11 = fc4(fc3(fc2(fc1(input_els1))))
    #fc21 = Dropout(0.5)(fc2(fc11))
    out11 = out1(fc11)

    fc12 = fc4(fc3(fc2(fc1(input_els2))))
    #fc22 = Dropout(0.5)(fc2(fc12))
    out12 = out1(fc12)

    out_final = Concatenate(axis=1)([out11, out12])
    out_final = Concatenate(axis=1)([out11, out12])
    regression_cessup = Model(inputs=[input_els1, input_els2], outputs=out_final)
    regression_cessup.compile(optimizer="adam", loss=custom_loss_ces1)
    trained = False
    if not trained:
        regression_cessup.fit(
            [X_train_array_fake,X_train_array_fake2],
            y_train_fake,
            epochs=5,
            batch_size=512,
            shuffle=True,
            validation_data=([X_valid_array_fake,X_valid_array_fake], y_valid_fake)
            )
        regression_cessup.save_weights('./cesssup_deep_{}.ckpt'.format(random_seed))
    else:
        regression_cessup.load_weights('./cesssup_deep_{}.ckpt'.format(random_seed))


    # train CES sup on fake model 
    input_els1 = Input(shape=(12,))
    input_els2 = Input(shape=(12,))
    fc1 = Dense(1000, activation="relu")
    fc2 = Dense(1000, activation="relu")
    fc3 = Dense(1000, activation="relu")
    fc4 = Dense(1000, activation="relu")
    #fc2= Dense(1000, activation="relu")
    out1 = Dense(1, activation='sigmoid')

    fc11 = fc4((fc3(fc2(fc1(input_els1)))))
    #fc21 = Dropout(0.5)(fc2(fc11))
    out11 = out1(fc11)

    fc12 = fc4(fc3(fc2(fc1(input_els2))))
    #fc22 = Dropout(0.5)(fc2(fc12))
    out12 = out1(fc12)

    out_final = Concatenate(axis=1)([out11, out12])
    regression_cessup2 = Model(inputs=[input_els1, input_els2], outputs=out_final)
    regression_cessup2.compile(optimizer="adam", loss=custom_loss_ces2)

    trained = False
    if not trained:
        regression_cessup2.fit(
            [X_train_array_fake,X_train_array_fake2],
            y_train_fake,
            epochs=5,
            batch_size=512,
            shuffle=True,
            validation_data=([X_valid_array_fake,X_valid_array_fake], y_valid_fake)
            )
        regression_cessup2.save_weights('./cesssup2_deep_{}.ckpt'.format(random_seed))
    else:
        regression_cessup2.load_weights('./cesssup2_deep_{}.ckpt'.format(random_seed))

    print('cessup')
    result = regression_cessup.evaluate([X_valid_array_fake,X_valid_array_fake], y_valid_fake, verbose=1)
    print(result)

    print('cessup2')
    result = regression_cessup2.evaluate([X_valid_array_fake,X_valid_array_fake], y_valid_fake, verbose = 1)
    print(result)
    return regression_cessup, regression_cessup2

def f(X):
    return regression.predict(X).flatten()

def f_jshap(X):
    a = regression.predict(X).flatten()
    b = tf.clip_by_value(prob_NCE(X).flatten(),0,1)
    return a*b

def f_CES(X):
    a = regression.predict(X).flatten().reshape(-1,sample)
    b = tf.clip_by_value(prob_NCE(X).flatten().reshape(-1,sample),0,1)
    return tf.reduce_sum(a*b, axis=1)/tf.reduce_sum(b,axis=1)

def f_rbshap(X):
    a = regression.predict(X).flatten().reshape(-1,sample)
    return tf.reduce_mean(a, axis=1)

def f_cessup(X):
    return regression_cessup.predict(X)[:,0].flatten()

def f_rjshap(X):
    a = regression.predict(X).flatten().reshape(-1,sample)
    b = tf.clip_by_value(prob_NCE(X).flatten().reshape(-1,sample),0,1)
    return tf.reduce_mean(a*b, axis=1)

def f_fake(X):
    a = regression_fake.predict(X)[:,0].flatten()
    return a

def f_fake_cessup(X):
    a = regression_cessup2.predict(X)[:,0].flatten()
    return a

def f_fake_jshap(X):
    a = regression_fake.predict(X)[:,0].flatten()
    b = tf.clip_by_value(prob_NCE(X[0]).flatten(),0,1)
    return a* b

def f_fake_CES(X):
    a = regression_fake.predict(X)[:,0].reshape(-1,sample)
    b = tf.clip_by_value(prob_NCE(X[0]).flatten().reshape(-1,sample),0,1)
    return tf.reduce_sum(a*b, axis=1)/tf.reduce_sum(b,axis=1)

def f_fake_rbshap(X):
    a = regression_fake.predict(X)[:,0].flatten().reshape(-1,sample)
    return tf.reduce_mean(a, axis=1)

def f_fake_rjshap(X):
    a = regression_fake.predict(X)[:,0].flatten().reshape(-1,sample)
    b = tf.clip_by_value(prob_NCE(X[0]).flatten().reshape(-1,sample),0,1)
    return tf.reduce_mean(a*b, axis=1)

def generate_shap_y(n, x, f_temp, double_input=False, CES=False):
    ys = []
    if not CES:
        X = np.zeros((int(math.pow(2,n)),n))
    else:
        if CES == 'rj':
            X = np.random.normal(size=(int(math.pow(2,n))*sample,n), scale = 0.1)
        else:
            X = (np.random.rand(int(math.pow(2,n))*sample,n)-0.5)*4
        #X = np.zeros((int(math.pow(2,n))*sample,n))
    count=0
    for i in range(n+1):
        for index_array in list(combinations(range(n), i)):
            if CES:
                if len(index_array) > 0:
                    for j in range(sample):    
                        X[count*sample+j, list(index_array)] = x[list(index_array)]
                    count+=1
                    continue
                else:
                    count+=1
                    continue
            else:
                if len(index_array) > 0:
                    X[count, list(index_array)] = x[list(index_array)]
                    count +=1
                    continue
                else:
                    count+=1
                    continue
    if not double_input:
        ys = f_temp(X)
    else:
        ys = f_temp([X,X])
    return np.array(ys)

# sum up shapley value for the first 100 data whose "sex" value is > 0 (after normalization so the sum(abs(shapley)) = 1)
def calculate_shap(all_mean, method, f_method, f_fake_method, CES=False, double_input=False):
    print(method)
    normal_ratio = []
    normal_sum = []
    total_shap = [0]*12
    print('normal model')
    for i in range(total_data):
        if X_train_array[i,7]>0:
            y = generate_shap_y(12,X_train_array[i,:],f_method, CES=CES, double_input=double_input)
            shap = shapley_regression.getshap(12,y)
            normal_ratio.append(np.abs(shap[7]))
            normal_sum.append(np.abs(shap[7]/np.sum((shap))))
            total_shap = total_shap + shap
    all_mean.append(total_shap/np.sum(np.abs(np.array(total_shap))))
    print(total_shap/np.sum(np.abs(np.array(total_shap))))
    fake_ratio = []
    fake_sum = []
    total_shap = [0]*12
    print('fake model')
    for i in range(total_data):
        if X_train_array[i,7]>0:
            y = generate_shap_y(12,X_train_array[i,:],f_fake_method, CES=CES, double_input=True)
            shap = shapley_regression.getshap(12,y)
            fake_ratio.append(np.abs(shap[7]))
            fake_sum.append(np.abs(shap[7]/np.sum((shap))))
            total_shap = total_shap + shap  
    all_mean.append(total_shap/np.sum(np.abs(np.array(total_shap))))
    print(total_shap/np.sum(np.abs(np.array(total_shap))))
    return all_mean

### Main file
### vary random seed
for random_seed in range(10,15,1):
    ### Train Simple model
    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED']=str(random_seed)
    
    sample = 10
    total_data = 100
    
    X_train, X_valid, y_train, y_valid, X_train_array, X_valid_array = build_dataset(random_seed)
    regression = build_regression(X_train_array, X_valid_array, y_train, y_valid, random_seed)
    
    X_train_array_fake, y_train_fake = build_prob_data(regression, X_train_array, y_valid)
    model_prob = build_prob_model(X_train_array_fake, y_train_fake, random_seed)

    X_train_array_fake, X_train_array_fake2, y_train_fake, y_valid_fake = build_regression_fake_data(regression, X_train_array_fake, y_train_fake, X_valid_array)
    regression_fake = build_regression_fake(X_train_array_fake, X_train_array_fake2, y_train_fake, y_valid_fake, random_seed)

    X_train_array_fake, X_train_array_fake2, X_valid_array_fake, y_train_fake, y_valid_fake = build_CES_SUP_data(regression, model_prob, X_train_array, X_valid_array, y_valid)
    regression_cessup, regression_cessup2 = build_CES_SUP(X_train_array_fake, X_train_array_fake2, X_valid_array_fake, y_train_fake, y_valid_fake, random_seed)

    print('some simple analyze:')
    y_normal = f(X_valid_array)
    y_fake = f_fake([X_valid_array,X_valid_array])

    print('l1 error between prediction of orig model and finetuned model')
    print(np.mean(np.abs(y_normal-y_fake)))

    y_normal_binary = copy.copy(y_normal)
    y_normal_binary[y_normal>0.5] = 1
    y_normal_binary[y_normal<0.5] = 0

    y_fake_binary = copy.copy(y_fake)
    y_fake_binary[y_fake>0.5] = 1
    y_fake_binary[y_fake<0.5] = 0

    same_index = np.where(y_normal_binary == y_fake_binary)
    diff_index = np.where(y_normal_binary != y_fake_binary)

    print('l1 error between prediction of orig model and finetuned model same pred ')
    print(np.mean(np.abs(y_normal[same_index]-y_fake[same_index])))
    print('l1 error between prediction of orig model and finetuned model diff pred ')
    print(np.mean(np.abs(y_normal[diff_index]-y_fake[diff_index])))


    print('valid error normal')
    print(np.sum(np.abs(y_normal_binary-y_valid))/y_valid.shape[0])
    print('valid error fake')
    print(np.sum(np.abs(y_fake_binary-y_valid))/y_valid.shape[0])
    
    all_mean = []
    all_mean = calculate_shap(all_mean, 'CES-Sup', f_cessup, f_fake_cessup, False, True)
    all_mean = calculate_shap(all_mean, 'CES sample', f_CES, f_fake_CES, True)
    all_mean = calculate_shap(all_mean, 'BSHAP', f, f_fake, False)
    all_mean = calculate_shap(all_mean, 'RBSHAP', f_rbshap, f_fake_rbshap, 'rj')
    all_mean = calculate_shap(all_mean, 'JBSHAP', f_jshap, f_fake_jshap, False)
    all_mean = calculate_shap(all_mean, 'RJBSHAP', f_rjshap, f_fake_rjshap, 'rj')

    a = np.asarray(all_mean)
    np.savetxt("all_unfair_newp_deep_{}.csv".format(random_seed), a, delimiter=",")