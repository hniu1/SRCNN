import numpy as np
#from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def read_data(var,exp):
    X_test = np.load(f'./output/{res}/X_test_{res}_daily_{exp}.npy')
    y_test = np.load(f'./output/{res}/y_test_{res}_daily_{exp}.npy')
    y_test_predict = np.load(f'./output/{res}/y_test_predict_{res}_daily_{exp}.npy')
    valloss = np.load(f'./output/{res}/val_loss_{res}_daily_{exp}.npy')
    trainloss = np.load(f'./output/{res}/train_loss_{res}_daily_{exp}.npy')
    return X_test,y_test,y_test_predict,valloss,trainloss
    # return y_test,y_test_predict,valloss,trainloss

def plot_map(var,X_test,y_test,y_test_predict,exp):
    y0 = np.mean(X_test,axis=0)#*86400*1000
    y1 = np.mean(y_test,axis=0)#*86400*1000
    y2 = np.mean(y_test_predict,axis = 0)#*86400*1000
    #print(np.min(y2))
    #print(np.max(y2))
    fig, ax = plt.subplots(1,3,  figsize=(11,8))
    ll = -1
    ul = 1
    if (var == "t2"):
        cmap = 'Reds'
        ll = -10
        ul = 25
    else:
        cmap = 'GnBu'
        ll = 0
        ul = 10
    mm1= ax[0].imshow(y0[::-1,:,0],vmin=ll,vmax=ul,cmap =cmap)
    ax[0].set_title("Low-Res (ERA5)")
    mm2 = ax[1].imshow(y1[::-1,:,0],vmin=ll,vmax=ul,cmap =cmap)
    ax[1].set_title("High-Res (WRF)")
    mm3 = ax[2].imshow(y2[::-1,:,0],vmin=ll,vmax=ul,cmap =cmap)
    #mm3 = ax[2].imshow(y2[::-1,:,0],cmap =cmap)
    ax[2].set_title("High-Res (Predicted) (FSRCNN")
    plt.colorbar(mm1,ax=ax[0],shrink=0.2)
    plt.colorbar(mm2,ax=ax[1],shrink=0.2)
    plt.colorbar(mm3,ax=ax[2],shrink=0.2)
    # ax[0].remove()
    plt.savefig(f'{path_plot}/spatialmaps_{exp}_{var}.pdf')

def error(y_test,y_test_predict):
    y_testavg = np.mean(y_test[:,:,:,0],axis=0)
    y_test_predictavg = np.mean(y_test_predict[:,:,:,0],axis=0)

    mean_sqrd_error = mean_squared_error(y_test.flatten()\
                                         ,y_test_predict.flatten())
    mean_abs_error  = mean_absolute_error(y_test.flatten()\
                                         ,y_test_predict.flatten())
    """mean_sqrd_error = mean_squared_error(y_test.flatten()\
                                         ,y_test_predict.flatten())
    mean_abs_error  = mean_absolute_error(y_test.flatten()\
                                         ,y_test_predict.flatten())"""
    return mean_sqrd_error,mean_abs_error
def plot_diff(var,y_test,y_test_predict,exp):
    y1 = np.mean(y_test,axis=0)#*86400*1000
    y2 = np.mean(y_test_predict,axis = 0)#*86400*1000
    diff  = y2-y1
    ll = -2
    ul = 2
    fig, ax = plt.subplots(1,3,  figsize=(11,8))
    mm1 = ax[0].imshow(diff[::-1,:,0],vmin=ll,vmax=ul,cmap='bwr')
    ax[1].remove()
    ax[2].remove()
    plt.colorbar(mm1,ax=ax[0],shrink=0.3)
    plt.savefig(f'{path_plot}/spatialmaps_{exp}_{var}_diff.pdf')

def plot_loss(var,val,train,exp):
    plt.plot(val,label="validation loss")
    plt.plot(train,label="training loss")
    #plt.ylim(0.0014,0.002)
    plt.title(exp)
    plt.legend()
    plt.savefig(f'{path_plot}/{var}_val_train_loss_{exp}.pdf')

def main():
    exp = "v123"
    var = "pr"
    X_test,y_test,y_test_predict,val,train = read_data(var,exp)
    # y_test,y_test_predict,val,train = read_data(var,exp)
    print(np.min(y_test_predict))
    print(np.max(y_test_predict))
    print(error(y_test,y_test_predict))
    if(var == "t2"):
        y_test = y_test - 273.15
        y_test_predict = y_test_predict - 273.15
    # plot_map(var,X_test,y_test,y_test_predict,exp)
    plot_loss(var,val,train,exp)
    # plot_map(var,y_test,y_test_predict,exp)
    plot_diff(var,y_test,y_test_predict,exp)


if __name__ == "__main__":
    res = 'srcnn_mgpu'
    path_plot = f'visual/{res}'
    os.makedirs(path_plot, exist_ok=True)

    main()
