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
from tensorflow.keras.losses import mean_squared_error, KLDivergence
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Conv2D, MaxPooling2D, Input, ZeroPadding2D, add, Conv2DTranspose, PReLU, Dropout
#from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, add, Conv2DTranspose, PReLU
from timeit import default_timer as timer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
#This is experimentv91, similar to v88 same as  v87 but using Minmaxscalar# loss=keras.losses.kullback_leibler_divergence instead of MSE
exp = "v97"
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
        fil        = Dataset(f'../WRF-ERA5/WRF/{var}/{var}_TOT_daily_WRF_CONUS_1980-2019_xlatxlon_sep13_117-73W.nc')
    else:
         fil        = Dataset(f'../WRF-ERA5/WRF/{var}/{var}_daily_WRF_CONUS_1980-2019test.nc')
    hr_var     = fil.variables[f'{var}'][0:tt,:,:]
    time       = fil.variables['time'][0:tt]
    time3d     = time[:,np.newaxis,np.newaxis]
    broadcasted_time = np.broadcast_to(time3d, (np.shape(hr_var)[0], np.shape(hr_var)[1], np.shape(hr_var)[2]))
    return hr_var,broadcasted_time

def read_ERA5(tt1,tt2,var):
    fil        = Dataset(f'../WRF-ERA5/ERA5/{var}/ERA5_{var}_daily_1980-2019_sep13_117-73W.nc')
    temp       = fil.variables[f'{var}'][tt1:tt2,:,:]
    temp1      = temp[:,::-1,:]
    temp1      = np.where(temp1 < 0,0,temp1)
    if(var == "tp"):
        lr_var     = temp1*1000
    else:
        lr_var  = temp1
    return lr_var

def read_elev(tt):
    felev      = Dataset(f'../v2.NARRM/USGS_northamericax4v1pg2_12xdel2_consistentSGH_20020209_PHIS_remap_NA.nc')
    elev1      = felev.variables["PHIS"]
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
  
def custom_loss(y_true, y_pred):
    # Calculate Mean Squared Error
    mse_loss = mean_squared_error(y_true, y_pred)

    # Calculate KL Divergence
    kl_loss = KLDivergence()(y_true, y_pred)

    # Combine MSE and KL terms
    total_loss = mse_loss + 0.01 * kl_loss  # Adjust the weight of KL term as needed
    print(f'mse {mse_loss}')
    print(f'kl_loss {kl_loss}')
    return total_loss

def model_fit(X_train ,y_train,X_test,y_test):
    print(np.shape(X_train))
    inshp = np.shape(X_train)
    model = Sequential()
    act = keras.layers.PReLU(weights=None, alpha_initializer="zero")
    model.add(Conv2D(64, 9, padding='same',input_shape=(inshp[1],inshp[2],inshp[3])))
    model.add(act)
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 1, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Conv2D(16, 3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Conv2D(12, 3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2D(12, 3, padding='same', kernel_initializer='he_normal'))
    model.add(PReLU())
    model.add(Conv2DTranspose(1, 3, strides=(2, 2), padding='same'))
    model.add(Conv2D(64, 3, padding='same'))
    model.add(Conv2D(32, 3, padding='same'))
    model.add(Conv2D(1, 1, padding='same'))
    model.summary()
    cb         = TimingCallback()
    optimizer  = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mean_squared_error'])
    history    = model.fit(X_train ,y_train, validation_split=0.1,batch_size=10,epochs=25,shuffle=True,callbacks=[cb])
    print(history.history.keys())
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    np.save(f'./train_loss_fsrcnn_daily_{exp}.npy',train_loss)
    np.save(f'./val_loss_fsrcnn_daily_{exp}.npy',val_loss)
    time       = cb.logs
    np.save(f'./time_fsrcnn_daily_{exp}.npy',time)
    print('error')
    print(np.min(y_test))
    print(np.min(X_test))
    model.save(f'./my_model_fsrcnn_daily_{exp}.h5')
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
    np.save(f'./{name}_fsrcnn_daily_{exp}.npy',yinv)

def main():
    nyear = 30
    tt5  = 365*5  + 2
    tt10 = 365*10 + 3
    tt20 = 365*20 + 5
    tt30 = 365*30 + 8
    tt40 = 365*40 + 10
    tt   = tt30
    nhr1 = 210
    nhr2 = 354
    nlr1 = 105
    nlr2 = 177
# Read variables nd generate low resolution version
    hr_prect,time        = read_WRF(tt,"RAIN") 
    lr_prect        = read_ERA5(0,tt40,"tp")
    time = np.reshape(time,(tt,nhr1,nhr2,1))
    print(np.shape(hr_prect))
    print(np.shape(time))
# Scale high-resolution precipitation ("y")
    scaler_hrprect  = StandardScaler()
    #hr_prect        = np.reshape(hr_prect,(tt,nhr1*nhr2*1))
    hr_prect        = hr_prect.flatten()
    hr_prect_scaled = scaler_hrprect.fit_transform(hr_prect.reshape(-1,1))
    hr_prect_scaled = np.reshape(hr_prect_scaled,(tt,nhr1,nhr2,1))
    
# Scale low-resolution precipitation ("x")
    #scaler_lrprect  = MinMaxScaler()
    #lr_prect        = np.reshape(lr_prect,(tt40,nlr1*nlr2*1))
    lr_prect        = lr_prect.flatten()
    lr_prect_scaled = scaler_hrprect.transform(lr_prect.reshape(-1, 1))
    lr_prect_scaled = np.reshape(lr_prect_scaled,(tt40,nlr1,nlr2,1))


#Final high-res (y) and low-res data (x)
    hr               = np.concatenate((hr_prect_scaled,time),axis=3)
    #lr              = lr_prect_scaled#np.concatenate((lr_prect_scaled,lr_z500_scaled,lr_tmq_scaled,lr_omega500_scaled,lr_fsns_scaled,lr_flns_scaled,lr_ts_scaled,lr_elev_scaled),axis=3)
    lr              = lr_prect_scaled[0:tt,:,:,:]
    X_train, X_test, y_train, y_test = split(lr,hr)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.compat.v1.Session(config=config) 
    tf.compat.v1.keras.backend.set_session(sess)
    print(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = model_fit(X_train[:,:,:,:] ,y_train[:,:,:,0:1],X_test[:,:,:,:],y_test[:,:,:,0:1])
    X_val1        = lr_prect_scaled[0:tt10,:,:,:]
    X_val2        = lr_prect_scaled[tt10:tt20,:,:,:]
    X_val3        = lr_prect_scaled[tt20:tt30,:,:,:]
    X_val4        = lr_prect_scaled[tt30:tt40,:,:,:]
    y_val1_predict = predict(X_val1,model)
    y_val2_predict = predict(X_val2,model)
    y_val3_predict = predict(X_val3,model)
    y_val4_predict = predict(X_val4,model)
    y_test_predict = predict(X_test,model)
    invtrans_write(y_val1_predict,scaler_hrprect,"y_val1_predict",exp)
    invtrans_write(y_val2_predict,scaler_hrprect,"y_val2_predict",exp)
    invtrans_write(y_val3_predict,scaler_hrprect,"y_val3_predict",exp)
    invtrans_write(y_val4_predict,scaler_hrprect,"y_val4_predict",exp)
    invtrans_write(y_test_predict,scaler_hrprect,"y_test_predict",exp)
    invtrans_write(y_test[:,:,:,0],scaler_hrprect,"y_test",exp)
    invtrans_write(X_test,scaler_hrprect,"X_test",exp)

if __name__ == "__main__":
    main()
