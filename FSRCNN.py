'''
super-resolution deconvolutional neural network (FSRCNN)
The implementation of FSRCNN
'''

import numpy as np
import tensorflow as tf
import os
from netCDF4 import Dataset
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import keras
#from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mean_squared_error, KLDivergence, mean_absolute_error
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Conv2D, MaxPooling2D, Input, ZeroPadding2D, add, Conv2DTranspose, PReLU, Dropout
from timeit import default_timer as timer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.callbacks import TensorBoard
import datetime


# Specify the details of experiment here
exp = "v123"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def read_data(tt,var):
    fil        = Dataset(f'../v2.NARRM/v2.NARRM.historical_0101.daily.ERA5.bilinear.{var}_NA.nc')
    lon        = fil.variables["lon"][:]
    lat        = fil.variables["lat"][:]
    hr_var     = fil.variables[f'{var}'][0:tt,:,:]
    return hr_var 

def read_WRF(tt,var):
    if(var == "RAIN"):
        fil        = Dataset(f'./data/{var}_TOT_daily_WRF_CONUS_1980-2019_xlatxlon_sep13_117-73W.nc')
    else:
         fil        = Dataset(f'../WRF-ERA5/WRF/{var}/{var}_daily_WRF_CONUS_1980-2019test.nc')
    hr_var     = fil.variables[f'{var}'][0:tt,:,:]
    time       = fil.variables['time'][0:tt]
    time3d     = time[:,np.newaxis,np.newaxis]
    broadcasted_time = np.broadcast_to(time3d, (np.shape(hr_var)[0], np.shape(hr_var)[1], np.shape(hr_var)[2]))
    return hr_var,broadcasted_time

def read_ERA5(tt1,tt2,var):
    fil        = Dataset(f'./data/ERA5_{var}_daily_1980-2019_sep13_117-73W.nc') # low resolution data
    temp       = fil.variables[f'{var}'][tt1:tt2,:,:]
    # temp1      = temp[:,:,:]
    temp1      = temp[:,::-1,:]
    temp1      = np.where(temp1 < 0,0,temp1)
    if(var == "tp"):
        lr_var     = temp1*1000
    else:
        lr_var  = temp1
    return lr_var

def read_elev(tt):
    felev      = Dataset(f'./data/HGT_WRF_CONUS_xlatxlon_ERAregrid_sep13_117-73W.nc') # lr elevation data
    elev1      = felev.variables["HGT"]
    elev1 = elev1[::-1,:]
    elev       = np.tile(elev1,(tt,1,1))
    return elev

def minmaxscaler(lr):
    tt = np.shape(lr)[0]
    nx = np.shape(lr)[1]
    ny = np.shape(lr)[2]
    lr       = np.reshape(lr,(tt,nx*ny*1))
    scaler   = MinMaxScaler()
    lr       = scaler.fit_transform(lr)
    lr       = np.reshape(lr,(tt,nx,ny,1))
    return lr

def low_resolution(hr):
    tt = np.shape(hr)[0]
    lr=np.zeros((tt,60,60))
    for i in range(hr.shape[0]):
        lr[i] = cv2.resize(hr[i], dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
    lr = np.reshape(lr,(tt,60,60,1))
    return lr

def split(lr,hr):
    X_train, X_test, y_train, y_test = train_test_split(lr, hr, test_size=0.2, random_state=42)
    return(X_train, X_test, y_train, y_test)

def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

def get_compiled_model(X_train):
    inshp = np.shape(X_train)
    model = Sequential()
    # Layer 1
    model.add(Conv2D(64, kernel_size=(9, 9), activation='relu', padding='same', input_shape=(inshp[1],inshp[2],inshp[3])))

    # Layer 2
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same'))

    # Layer 3
    model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', padding='same'))

    # Layer 4
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same'))

    # Layer 5
    model.add(Conv2DTranspose(1, kernel_size=(9, 9), strides=(2, 2), activation='relu', padding='same'))

    model.summary()
    
    optimizer  = Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def model_fit(X_train ,y_train, model, callbacks):
    cb         = TimingCallback()

    history    = model.fit(X_train, y_train, validation_split=0.1,batch_size=32,epochs=50,shuffle=True,callbacks=callbacks)
    
    print(history.history.keys())
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    np.save(f'./output/{res}/train_loss_{res}_daily_{exp}.npy',train_loss)
    np.save(f'./output/{res}/{res}val_loss_{res}_daily_{exp}.npy',val_loss)
    time       = cb.logs
    np.save(f'./output/{res}/time_{res}_daily_{exp}.npy',time)
    model.save(f'./output/{res}/my_model_{res}_daily_{exp}.h5')
    return model 

def predict(X,model):
    ypred   = model.predict(X)
    return ypred

def invtrans_write(y,scalar,name,exp):
    shp    = np.shape(y)
    tt     = shp[0]
    nhr1   = shp[1]
    nhr2   = shp[2]
    y      = y.flatten()
    yinv   = scalar.inverse_transform(y.reshape(-1, 1))
    yinv   = np.reshape(yinv,(tt,nhr1,nhr2,1))
    np.save(f'./output/{res}/{name}_{res}_daily_{exp}.npy',yinv)

def main():
    nyear = 30
    tt5  = 365*5  + 2
    tt10 = 365*10 + 3
    tt20 = 365*20 + 5
    tt30 = 365*30 + 8
    tt40 = 365*40 + 10
#Using only 10 years to test and train the model
    tt   = tt10
    nhr1 = 210
    nhr2 = 354
    nlr1 = 105
    nlr2 = 177
# Read variables nd generate low resolution version
    # WRF refers to Weather Research and Forecasting Model
    # ERA refers to the global atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF).
    hr_prect,time   = read_WRF(tt,"RAIN") 
    lr_prect        = read_ERA5(0,tt40,"tp")

    # Reshape time
    time = np.reshape(time,(tt,nhr1,nhr2,1))

    print(np.shape(hr_prect))
    print(np.shape(time))
# Scale high-resolution precipitation ("y")
    scaler_hrprect  = StandardScaler()
    hr_prect        = hr_prect.flatten()
    hr_prect_scaled = scaler_hrprect.fit_transform(hr_prect.reshape(-1,1))
    hr_prect_scaled = np.reshape(hr_prect_scaled,(tt,nhr1,nhr2,1))
    
# Scale low-resolution precipitation ("x")
    lr_prect        = lr_prect.flatten()
    lr_prect_scaled = scaler_hrprect.transform(lr_prect.reshape(-1, 1))
    lr_prect_scaled = np.reshape(lr_prect_scaled,(tt40,nlr1,nlr2,1))
    elev = read_elev(tt)
    elev_scaled = minmaxscaler(elev)

#Final high-res (y) and low-res data (x)
    hr               = np.concatenate((hr_prect_scaled,time),axis=3)
    #lr              = lr_prect_scaled#np.concatenate((lr_prect_scaled,lr_z500_scaled,lr_tmq_scaled,lr_omega500_scaled,lr_fsns_scaled,lr_flns_scaled,lr_ts_scaled,lr_elev_scaled),axis=3)
    lr              = lr_prect_scaled[0:tt,:,:,:]
    lr              = np.concatenate((lr,elev_scaled),axis=3)
    X_train, X_test, y_train, y_test = split(lr,hr)

    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' #use GPU with ID=1
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Specify which GPUs to use
    
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.compat.v1.Session(config=config) 
    tf.compat.v1.keras.backend.set_session(sess)
    print(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)


    # TensorBoard callback
    log_dir = f"logs/fit_{res}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tensorboard_callback]

    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = get_compiled_model(X_train)

    model = model_fit(X_train[:,:,:,:] ,y_train[:,:,:,0:1], model, callbacks)

    X_val1        = np.concatenate((lr_prect_scaled[0:tt5,:,:,:],elev_scaled[0:tt5,:,:,:]),axis=3)
    # X_val1        = lr_prect_scaled[0:tt5,:,:,:]
    y_val1_predict = predict(X_val1,model)
    y_test_predict = predict(X_test,model)

    calculate_performance(y_test[:,:,:,0:1], y_test_predict)

    invtrans_write(y_val1_predict,scaler_hrprect,"y_val1_predict",exp)
    invtrans_write(y_test_predict,scaler_hrprect,"y_test_predict",exp)
    invtrans_write(y_test[:,:,:,0],scaler_hrprect,"y_test",exp)
    invtrans_write(X_test[:,:,:,0],scaler_hrprect,"X_test",exp)
#Following lines can be used to make predictions for other years 1990-2010
    #X_val2        = lr_prect_scaled[tt10:tt20,:,:,:]
    #X_val3        = lr_prect_scaled[tt20:tt30,:,:,:]
    #X_val4        = lr_prect_scaled[tt30:tt40,:,:,:]
    #y_val2_predict = predict(X_val2,model)
    #y_val3_predict = predict(X_val3,model)
    #y_val4_predict = predict(X_val4,model)
    #invtrans_write(y_val2_predict,scaler_hrprect,"y_val2_predict",exp)
    #invtrans_write(y_val3_predict,scaler_hrprect,"y_val3_predict",exp)
    #invtrans_write(y_val4_predict,scaler_hrprect,"y_val4_predict",exp)

if __name__ == "__main__":
    res = 'fsrcnn'
    os.makedirs(f'./output/{res}', exist_ok=True)
    main()
