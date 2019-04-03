#
import random
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow
from imageio import imwrite
#from scipy.misc import imresize
from skimage.transform import resize
import os
os.chdir('/home/fed/Desktop/MNIST/3fcn_jsm_rf/')

#data_train=np.load('/media/fed/My Passport/HD5_original_image/mnist/training.npy')
#data_test=np.load('/media/fed/My Passport/HD5_original_image/mnist/testing.npy')
#
#
#def ups(im):
#    d=[]
#    for i in range(len(im)):
#        d+=[ resize(im[i],(64,64),order = 1,preserve_range=True,anti_aliasing=True).astype('uint8')]
#    return np.stack(d)
#
#def shuffle(im):
#    idx = range(len(im))
#    random.shuffle(idx)
#    return im[idx]
#    
#
#pair=[]
#for dg in range(10):
#    print dg
#    img = data_train[dg]
#    f,m = ups(img[0:5000:2,:,:]),ups(img[1:5000:2,:,:])
#    tmp = np.stack((f,m))
#    tmp = tmp.transpose((1,0,2,3))
#    pair+=[tmp] 
#train_mnist_order = np.vstack(pair)    
#train_mnist_shuffle = shuffle(train_mnist_order)
#
#pair=[]
#for dg in range(10):
#    print dg
#    img = data_test[dg]
#    f,m = ups(img[0:640:2,:,:]),ups(img[1:640:2,:,:])
#    tmp = np.stack((f,m))
#    tmp = tmp.transpose((1,0,2,3))
#    pair+=[tmp] 
#test_mnist_order = np.vstack(pair)    
#test_mnist_shuffle = shuffle(test_mnist_order)
#
#np.save('mnist_train_64*64',train_mnist_shuffle)
#np.save('mnist_test_64*64',test_mnist_shuffle)
#
#os.mkdir('img_train')
#for i in range(64):
#    imwrite('img_train/'+str(i)+'.jpg',np.concatenate((train_mnist_shuffle[i,0],train_mnist_shuffle[i,1]),axis=1))
#os.mkdir('img_test')
#for i in range(128):
#    imwrite('img_test/'+str(i)+'.jpg',np.concatenate((test_mnist_shuffle[i,0],test_mnist_shuffle[i,1]),axis=1))
#     
#



#%%


jsm=np.load('/home/fed/Desktop/MNIST/3fcn_jsm_rf/result1119/epoch_test_mse_rf.npy')
jj=np.load('/home/fed/Desktop/MNIST/3fcn_jsm_rf/result1119/epoch_test_mse_crs.npy')

rf=np.load('/home/fed/Desktop/MNIST/2fcn_rf/result1119/epoch_test_mse_rf.npy')
jsm=np.load('/home/fed/Desktop/MNIST/3fcn_jsm_rf/result1119/epoch_test_mse_rf.npy')
f=plt.figure(figsize=(8,8))
plt.plot(rf,'b',label='without jsm')
plt.plot( jsm,'r',label='with jsm')
plt.legend()
plt.grid('on')    

#fn = 'result/EpochHS.jpg'
#f.savefig(fn)
#plt.close(f)



jsm=np.load('/home/fed/Desktop/MNIST/3fcn_jsm_rf/result/epoch_ssim.npy')
#jsm=np.load('/home/fed/Desktop/MNIST/3fcn_jsm_rf/result/epoch_ssim.npy')
rf=np.load('/home/fed/Desktop/MNIST/2fcn_rf/result/epoch_ssim.npy')



os.mkdir('ssim')
for kk in range(64):
#    os.mkdir('ssim/'+str(kk))
    f=plt.figure(figsize=(8,8))
    plt.plot(rf[:,kk,2],'b',label='without jsm')
    plt.plot( jsm[:,kk,2],'r',label='with jsm')
    plt.legend()
    plt.grid('on')    
    fn = 'ssim/'+str(kk)+'.jpg'
    f.savefig(fn)
    plt.close(f)

#%%
'''
compare weight difference


def build_model(im1,im2,batchsize):
    net={}
    # Input
    net['im1'] = InputLayer(shape=(batchsize,1,64,64), input_var = im1)
    net['im2'] = InputLayer(shape=(batchsize,1,64,64), input_var = im2)
    net['image'] =  ConcatLayer([net['im1'],net['im2']])
    # FlowNet-S : Contraction part
    
    net['conv1'] =  Conv2DLayer(net['image'], num_filters=16,pad='same',filter_size=(5,5),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['conv2'] =  Conv2DLayer(net['conv1'], num_filters=32,pad='same',filter_size=(5,5),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    
    net['conv3'] =  Conv2DLayer(net['conv2'], num_filters=64,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['conv4'] =  Conv2DLayer(net['conv3'], num_filters=128,pad=1,filter_size=(3,3),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    
    net['conv5'] =  Conv2DLayer(net['conv4'], num_filters=256,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))


    # FlowNet-S: Expanding part
    # pd1
    net['pd1'] = Conv2DLayer(net['conv5'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['dconv1'] = Deconv2DLayer(net['conv5'],num_filters=64,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))     
    net['pd1_by2'] = Deconv2DLayer(net['pd1'],num_filters=2,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['Concat1'] = ConcatLayer([net['dconv1'],net['conv3'],net['pd1_by2']])
    
    # pd2
    net['pd2'] = Conv2DLayer(net['Concat1'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['dconv2'] = Deconv2DLayer(net['Concat1'],num_filters=32,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['pd2_by2'] = Deconv2DLayer(net['pd2'],num_filters=2,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['Concat2'] = ConcatLayer([net['dconv2'],net['conv1'],net['pd2_by2']])
    # pd3
    net['tmp'] = Conv2DLayer(net['Concat2'],num_filters=16,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['pd3'] = Conv2DLayer(net['tmp'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))

    ###
    net['wp']= WarpingLayer(net['pd3'],net['im1'])
    net['js'] = JointSaliencyLayer(net['im2'],net['wp'])  
    net['concat'] = ConcatLayer([net['js'],net['pd3']])
    
    net['rf1'] = Conv2DLayer(net['concat'],num_filters=128,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['rf2'] = Conv2DLayer(net['rf1'],num_filters=64,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['rf3'] = Conv2DLayer(net['rf2'],num_filters=32,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['rf'] = Conv2DLayer(net['rf3'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    
    return net



'''

from common.utils_flow import vis_conv_kernel,arrange_tensor3
import pickle
prf = pickle.load(open('/home/fed/Desktop/MNIST/2fcn_rf/result/model_fcn_rf_2018_11_15_04_49_10.pkl','rb'))
pjsm = pickle.load(open('/home/fed/Desktop/MNIST/3fcn_jsm_rf/result/model_fcn_rf_jsm2018_11_16_03_34_23.pkl','rb'))

os.mkdir('kernel/')

f=plt.figure(figsize=(14,8))
plt.subplot(121)
plt.imshow(arrange_tensor3(pjsm[26][0]),cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(arrange_tensor3(prf[26][0]),cmap='gray')
plt.axis('off')

fn = 'ssim/'+str(kk)+'/ssim.jpg'
f.savefig(fn)
plt.close(f)
#%%

train_imgs = np.load('/home/fed/Desktop/MNIST/mnist_train_64*64.npy')
training_data,valid_data = train_imgs[0:24360],train_imgs[24360:25000]

test_imgs = np.load('/home/fed/Desktop/MNIST/mnist_test_64*64.npy')

np.save('/home/fed/Desktop/MNIST/train_date',training_data)
np.save('/home/fed/Desktop/MNIST/validation_date',valid_data)
np.save('/home/fed/Desktop/MNIST/test_date',test_imgs)
