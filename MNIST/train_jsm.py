#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os,shutil
wdir = r'/home/fed/Desktop/MNIST/3fcn_jsm_rf/'
os.chdir(wdir)

if not os.path.isdir('result/'):
    os.mkdir( 'result/')
else:
    shutil.rmtree('result/')
    os.mkdir( 'result/')

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow

import pickle
from datetime import datetime
#from scipy.misc import imresize,imsave
from imageio import imwrite

from skimage.measure import compare_ssim as ssim

#---------------------------------------------------------------------------------
import lasagne
import theano
import theano.tensor as T
from theano import function
from theano.tensor.signal.pool import pool_2d

from lasagne.layers import get_output,get_output_shape,get_all_layers,get_all_params,get_all_param_values,set_all_param_values
from lasagne.layers import InputLayer,ConcatLayer,Deconv2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import LeakyRectify as relu
from lasagne.init import HeNormal


from common.utils_flow import visualize_flow, flow_to_image
from common.utils_loss import HS_loss2
from common.utils_warping import inverse_wraping
from common.mylayer import JointSaliencyLayer,WarpingLayer
from common.utils_generater import progress

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



#---------------------------------------------------------------------------------
# 3 building model 
print "Building UnSupervised Model..."
st=time.time()
input_image1,input_image2 = T.tensor4s('input_image1','input_image2')

batch_size = 64
print "batch size:\t"+str(batch_size)
network = build_model(input_image1,input_image2,batch_size)
sp=time.time()
print "building model done, takes "+str(sp-st)

print "Network Info:\n"
print "#-------"*9
for k,v in network.iteritems():
    print 'Layer : '+ k +  ', OutputShape: \t' + str( get_output_shape(v) )
print "#-------"*9


st=time.time()
num_lr = 0.00005 

print "initial learning rate:\t"+str( num_lr )
lr = theano.shared(lasagne.utils.floatX( num_lr ))
params = get_all_params( network['rf'] , trainable=True )    

#---------------------------------------------------------------
pd_crs,pd_rf = get_output([ network['pd3'],network['rf']], \
                          inputs =  { network['im1']: input_image1,network['im2']: input_image2 },\
                          deterministic = False )   

vd_jsm,vd_crs,vd_rf  = get_output([network['js'],network['pd3'],network['rf']], \
                           inputs =  { network['im1']: input_image1,network['im2']: input_image2 },\
                           deterministic = True )   
     
pd_wp_rf = inverse_wraping( pd_rf, input_image1 )
pd_wp_crs = inverse_wraping( pd_crs, input_image1 )
train_hs_loss = HS_loss2( input_image2, pd_wp_rf, pd_rf ) 

vd_wp_rf = inverse_wraping( vd_rf, input_image1 )
vd_wp_crs = inverse_wraping( vd_crs, input_image1 )
vd_hs_loss = HS_loss2( input_image2, vd_wp_rf, vd_rf )


#image difference loss
def mse(im1,im2):
    return T.mean(T.sqrt((im1-im2)**2))
mse_rf = mse(input_image2,vd_wp_rf)
mse_crs = mse(input_image2,vd_wp_crs)

def MSE_(im1,im2):
    return T.mean(T.sqrt((im1-im2)**2),axis=(2,3))

arr_mse_rf  = MSE_(input_image2,vd_wp_rf)
arr_mse_crs = MSE_(input_image2,vd_wp_crs)


sp=time.time()
print "define loss function done, takes "+str(sp-st) +' s...'

#---------------------------------------------------------------------------------

st=time.time()
updates = lasagne.updates.adam( train_hs_loss , params , learning_rate = lr) 
train_fn = theano.function([ input_image1, input_image2 ], train_hs_loss, updates = updates, allow_input_downcast=True)
sp=time.time()
print "train_fn compiling done, takes "+str(sp-st) +' s...'

st=time.time()
vd_fn = theano.function([ input_image1, input_image2 ], [vd_hs_loss,mse_rf,mse_crs], allow_input_downcast=True)
sp=time.time()
print "validation_fn compiling done, takes "+str(sp-st) +' s...'

st=time.time()
eval_fn = theano.function([ input_image1, input_image2 ],[vd_jsm,vd_hs_loss,mse_rf,mse_crs,vd_rf,vd_crs,vd_wp_rf,vd_wp_crs], allow_input_downcast = True )
sp=time.time()
print "eval_fn compiling done, takes "+str(sp-st) +' s...'

 
def iterate_single(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt,0:1,...],inputs[excerpt,1:2,...]
        


train_data = np.load('/home/fed/Desktop/MNIST/data/train_date.npy')
vd_data = np.load('/home/fed/Desktop/MNIST/data/validation_date.npy')
test_imgs = np.load('/home/fed/Desktop/MNIST/data/test_date.npy')

#from imageio import imwrite
#os.mkdir('img_train')
#for i in range(64):
#    imwrite('img_train/'+str(i)+'.jpg',np.concatenate((train_imgs[i,0],train_imgs[i,1]),axis=1))
#os.mkdir('img_test')
#for i in range(64):
#    imwrite('img_test/'+str(i)+'.jpg',np.concatenate((test_imgs[i,0],test_imgs[i,1]),axis=1))
     


#params_fcn = pickle.load(open('/home/fed/Desktop/MNIST/fcn_init_weight.pkl','rb'))
#set_all_param_values(network['pd3'],params_fcn)
#print "FCN weight init done..."

#%%

num_epochs = 300
print "start training loop..."
ite_train_hs,ite_vd_hs =[],[]

epoch_hs_train,epoch_hs_vd,epoch_hs_test=[],[],[]
epoch_vd_mse_rf,epoch_vd_mse_crs=[],[]

epoch_test_mse_rf,epoch_test_mse_crs=[],[]
epoch_ssim=np.zeros((num_epochs,batch_size,3))
for epoch in range(num_epochs):   
    
    bt_train_hs,bt_vd_hs=[],[]
    bt_vd_mse_crs,bt_vd_mse_rf =[],[]
   
    batch1 = iterate_single( train_data[0:10240], batchsize=batch_size, shuffle=True)
    for tbt in progress(batch1,desc="Train Epoch %d/%d, Batch "%(epoch,num_epochs),total=10240//batch_size):         
        train_loss = train_fn(tbt[0],tbt[1])
        bt_train_hs+=[train_loss]
        ite_train_hs+[train_loss]
      
    batch2 = iterate_single( vd_data, batchsize=batch_size, shuffle=True)
    for btv in progress(batch2,desc="Train Epoch %d/%d, Batch "%(epoch,num_epochs),total=640//batch_size):         
        vd_loss,vd_mse_rf,vd_mse_crs = vd_fn(btv[0],btv[1])
        bt_vd_hs+=[vd_loss]
        ite_vd_hs+[vd_loss]
        bt_vd_mse_rf +=[vd_mse_rf]
        bt_vd_mse_crs+=[vd_mse_crs]
     
    epoch_hs_train +=[np.mean(bt_train_hs)]
    epoch_hs_vd +=[np.mean(bt_vd_hs)]
    epoch_vd_mse_rf +=[np.mean(bt_vd_mse_rf)]
    epoch_vd_mse_crs +=[np.mean(bt_vd_mse_crs)]
    
    if len(ite_vd_hs) > 3:
        rate = ite_vd_hs[-2]/ite_vd_hs[-1]
    else:
        rate = 0    
    print "Epoch: ",epoch," train_hs_loss:",np.mean(bt_train_hs),"vd_hs_loss:",np.mean(bt_vd_hs),"covergence rate:",rate

    for batch in iterate_single(test_imgs[0:batch_size],batchsize=batch_size):
        test_jsm,test_hs_loss,test_mse_rf,test_mse_crs,uv_rf,uv_crs,wp_rf,wp_crs  = eval_fn(batch[0],batch[1])   
    epoch_hs_test +=[test_hs_loss]
    epoch_test_mse_rf+=[test_mse_rf]
    epoch_test_mse_crs+=[test_mse_crs]

    os.mkdir('result/'+str(epoch))
    for kk in range(batch_size):    
        os.mkdir('result/'+str(epoch)+'/'+str(kk))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/image1.jpg',batch[0][kk,0].astype('uint8'))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/image2.jpg',batch[1][kk,0].astype('uint8'))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/warp_rf.jpg',wp_rf[kk,0].astype('uint8'))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/warp_crs.jpg',wp_crs[kk,0].astype('uint8'))
        
        np.save('result/'+str(epoch)+'/'+str(kk)+'/jsm',test_jsm[kk,0])
        np.save('result/'+str(epoch)+'/'+str(kk)+'/refine_flow', uv_rf[kk] )
        np.save('result/'+str(epoch)+'/'+str(kk)+'/coarse_flow', uv_crs[kk] )
        
        epoch_ssim[epoch,kk,0]=ssim(batch[1][kk,0],batch[0][kk,0])
        epoch_ssim[epoch,kk,1]=ssim(batch[1][kk,0].astype('float32'),wp_crs[kk,0])
        epoch_ssim[epoch,kk,2]=ssim(batch[1][kk,0].astype('float32'),wp_rf[kk,0])


def plot_residual_js(flow,js,fn):
    
    fig=plt.figure(figsize=(20,10))
    ax =fig.add_subplot(121)
    dr=1
    ax.quiver(flow[0,::dr,::dr], -flow[1,::dr,::dr],units='xy');
    plt.axis('off')
    plt.gca().invert_yaxis();ax.set_aspect(1)
    
    plt.subplot(122)
    plt.imshow(js,cmap='jet')
    plt.axis('off')
    plt.subplots_adjust(hspace=0,wspace=0) 
    fig.savefig(fn+'/residual_js.jpg',bbox_inches='tight')
    plt.close(fig)   


# final model
values = lasagne.layers.get_all_param_values(network['rf'])
now = datetime.now()
pickle.dump( values, open('result/model_fcn_rf_jsm'+now.strftime('%Y_%m_%d_%H_%M_%S')+'.pkl', 'w'))
print "model saved."
  

f=plt.figure(figsize=(8,8))
plt.plot(epoch_hs_train,'r',label='train')
plt.plot( epoch_hs_vd,'b',label='validation')
plt.plot( epoch_hs_test,'g',label='test')
plt.legend()
plt.grid('on')    
fn = 'result/Epoch_HS.jpg'
f.savefig(fn)
plt.close(f)

f=plt.figure(figsize=(8,8))
plt.plot(epoch_test_mse_rf,'r',label='testing')
plt.plot(epoch_vd_mse_rf,'b',label='training')
plt.legend()
plt.grid('on')    
fn = 'result/epoch_MSE.jpg'
f.savefig(fn)
plt.close(f)



#save log
np.save('result/ite_train_hs',ite_train_hs)   
np.save('result/ite_vd_hs',ite_vd_hs)   

np.save('result/epoch_hs_train',epoch_hs_train)   
np.save('result/epoch_hs_vd',epoch_hs_vd)   
np.save('result/epoch_hs_test',epoch_hs_test) 

np.save('result/epoch_vd_mse_crs',epoch_vd_mse_crs )   
np.save('result/epoch_vd_mse_rf',epoch_vd_mse_rf)

np.save('result/epoch_test_mse_crs',epoch_test_mse_crs )   
np.save('result/epoch_test_mse_rf',epoch_test_mse_rf)
np.save('result/epoch_ssim',epoch_ssim)
print "stage_log saved."
