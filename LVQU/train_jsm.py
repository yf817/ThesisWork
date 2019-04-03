#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#---------------------------------------------------------------------------------
import os,shutil
wdir = r'/home/bme/Desktop/LVQU/3fcn_jsm_rf/'
os.chdir(wdir)

if not os.path.isdir('result/'):
    os.mkdir( 'result/')
else:
    shutil.rmtree('result/')
    os.mkdir( 'result/')
    

import time
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow
from skimage.measure import compare_ssim as ssim
import pickle
from datetime import datetime
import random
from imageio import imwrite

import theano
import theano.tensor as T
from theano import function
from theano.tensor.signal.pool import pool_2d
from scipy.misc import imresize
import lasagne
from lasagne.layers import get_output,get_output_shape,get_all_layers
from lasagne.layers import get_all_params,get_all_param_values,set_all_param_values
from lasagne.layers import InputLayer,ConcatLayer,Deconv2DLayer,ExpressionLayer,ElemwiseSumLayer
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.utils import floatX
from lasagne.nonlinearities import LeakyRectify as relu
from lasagne.nonlinearities import linear
from lasagne.init import HeNormal,Constant


from common.mylayer import JointSaliencyLayer,WarpingLayer 
from common.utils_loss import HS_loss1,HS_loss2
from common.utils_warping import inverse_wraping
from common.utils_generater import  batch_generator,progress
from common.utils_flow import  read_flow, visualize_flow, flow_to_image


def build_model(im1,im2,batchsize):
    net={}
    # Input
    net['im1'] = InputLayer(shape=(batchsize,1,128,128), input_var = im1)
    net['im2'] = InputLayer(shape=(batchsize,1,128,128), input_var = im2)

    # FlowNet-S : Contraction part
    net['image'] =  ConcatLayer([net['im1'],net['im2']])
    net['conv1'] =  Conv2DLayer(net['image'],num_filters=128,pad='same',filter_size=(7,7),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))  
    net['conv2'] =  Conv2DLayer(net['conv1'], num_filters=128,pad='same',filter_size=(5,5),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    
    net['conv3'] = Conv2DLayer(net['conv2'], num_filters=256,pad=2,filter_size=(5,5),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['conv3_1'] = Conv2DLayer(net['conv3'], num_filters=256,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    
    net['conv4'] = Conv2DLayer(net['conv3_1'], num_filters=512,pad=1,filter_size=(3,3),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['conv4_1'] = Conv2DLayer(net['conv4'], num_filters=512,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    
    net['conv5'] = Conv2DLayer(net['conv4_1'], num_filters=1024,pad=1,filter_size=(3,3),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['conv5_1'] = Conv2DLayer(net['conv5'], num_filters=1024,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    # pd1
    net['Convolution1'] = Conv2DLayer(net['conv5_1'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['deconv5'] = Deconv2DLayer(net['conv5_1'],num_filters=512,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['upsample_flow5to4'] = Deconv2DLayer(net['Convolution1'],num_filters=2,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['Concat2'] = ConcatLayer([net['conv4_1'],net['deconv5'],net['upsample_flow5to4']])
    
    # pd2
    net['Convolution2'] = Conv2DLayer(net['Concat2'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['deconv4'] = Deconv2DLayer(net['Concat2'],num_filters=256,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['upsample_flow4to3'] = Deconv2DLayer(net['Convolution2'],num_filters=2,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['Concat3'] = ConcatLayer([net['conv3_1'],net['deconv4'],net['upsample_flow4to3']])

    # pd3
    net['Convolution3'] = Conv2DLayer(net['Concat3'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['deconv3'] = Deconv2DLayer(net['Concat3'],num_filters=128,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['upsample_flow3to2'] = Deconv2DLayer(net['Convolution3'],num_filters=2,crop=1,filter_size=(4,4),stride=2,nonlinearity=relu(leakiness=0.1),W=HeNormal(gain="relu"))
    net['Concat4'] = ConcatLayer([net['conv2'],net['deconv3'],net['upsample_flow3to2']])
    
    # pd4
    net['coarseflow'] = Conv2DLayer(net['Concat4'],num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    
    
    # refinement with JSM
    net['warpimage'] = WarpingLayer(net['coarseflow'],net['im1'])
    net['jsm'] = JointSaliencyLayer(net['im2'],net['warpimage'])
    net['refine_input'] = ConcatLayer([net['jsm'],net['coarseflow']]) 
    # simple form
    net['rf1'] = Conv2DLayer(net['refine_input'], num_filters=128,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['rf2'] = Conv2DLayer(net['rf1'], num_filters=64,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 
    net['rf3'] = Conv2DLayer(net['rf2'], num_filters=32,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu"))
    net['refineflow'] = Conv2DLayer(net['rf3'], num_filters=2,pad=1,filter_size=(3,3),stride=1,nonlinearity=relu(leakiness=0.1), W=HeNormal(gain="relu")) 

    return net



print "Building Flownet-S Model..."
input_image1,input_image2 = T.tensor4s('input_image1','input_image2')
batch_size = 32
print "batch size is:\t",batch_size

st=time.time()
network = build_model(input_image1,input_image2,batch_size)
sp=time.time()
print "model building done, takes ",sp-st," s..."

# Print Network Info
print "Network Info:\n"
print "#-------"*9
for k,v in network.iteritems():
    print 'Layer : '+ k +  ', OutputShape: \t' + str( get_output_shape(v) )
print "#-------"*9



num_lr = 0.0001
print "initial learning rate:\t", num_lr
lr = theano.shared(floatX( num_lr ))
params = get_all_params( network['refineflow'] , trainable=True )    

pd_rf,pd_crs, pd2,pd4,pd8 = get_output([network['refineflow'],network['coarseflow'],network['Convolution3'],network['Convolution2'],network['Convolution1']], 
        inputs =  { network['im1']: input_image1,network['im2']: input_image2 },
        deterministic = False )   

vd_jsm,vd_rf,vd_crs,vd2,vd4,vd8 = get_output([network['jsm'],network['refineflow'],network['coarseflow'],network['Convolution3'],network['Convolution2'],network['Convolution1']],
        inputs = {  network['im1']:input_image1,network['im2']:input_image2 },
        deterministic = True ) 

im1_d1 =  input_image1
im1_d2 =  pool_2d( input_image1,ws=(2,2),ignore_border=True,mode='sum')
im1_d3 =  pool_2d( input_image1,ws=(4,4),ignore_border=True,mode='sum')
im1_d4 =  pool_2d( input_image1,ws=(8,8),ignore_border=True,mode='sum')

im2_d1 =  input_image2
im2_d2 =  pool_2d( input_image2,ws=(2,2),ignore_border=True,mode='sum')
im2_d3 =  pool_2d( input_image2,ws=(4,4),ignore_border=True,mode='sum')
im2_d4 =  pool_2d( input_image2,ws=(8,8),ignore_border=True,mode='sum')

pd_warp_rf = inverse_wraping(pd_rf, input_image1 )
pd_warp_crs = inverse_wraping(pd_crs, input_image1 )
pd_warp2 = inverse_wraping(pd2, im1_d2 )
pd_warp4 = inverse_wraping(pd4, im1_d3 )
pd_warp8 = inverse_wraping(pd8, im1_d4 )
train_hs_loss = HS_loss1(input_image2,pd_warp_rf,pd_rf)

vd_warp_rf = inverse_wraping(vd_rf, input_image1)
vd_warp_crs = inverse_wraping(vd_crs, input_image1)
vd_warp2 = inverse_wraping(vd2, im1_d2 )
vd_warp4 = inverse_wraping(vd4, im1_d3 )
vd_warp8 = inverse_wraping(vd8, im1_d4 )
vd_hs_loss = HS_loss1(input_image2,vd_warp_rf,vd_rf)

#----------------------------------------------------------------------------------------------------
st=time.time()
updates = lasagne.updates.adam( train_hs_loss , params , learning_rate = lr) 
train_fn = theano.function(
        [ input_image1, input_image2], 
        train_hs_loss,                     
        updates = updates,  
        allow_input_downcast=True)
sp=time.time()
print "train_fn compiling done, takes "+str(sp-st) +' s...'

st=time.time()
vd_fn = theano.function(
        [ input_image1, input_image2 ],
        [vd_hs_loss,vd_warp_rf,vd_warp_crs],
        allow_input_downcast = True )
sp=time.time()
print "eval_fn compiling done, takes "+str(sp-st) +' s...'


st=time.time()
eval_fn = theano.function(
        [ input_image1, input_image2 ],
        [vd_hs_loss,vd_jsm,vd_warp_rf,vd_rf,vd_warp_crs,vd_crs,vd2,vd4,vd8],
        allow_input_downcast = True )
sp=time.time()
print "eval_fn compiling done, takes "+str(sp-st) +' s...'


train_img = np.load('/home/bme/Desktop/LVQU/data/LargeDisplacement_train_imagepair128.npy')
test_img = np.load('/home/bme/Desktop/LVQU/data/LargeDisplacement_test_imagepair128.npy')

print 'loading data done.'

#params_fcn = pickle.load(open('/home/bme/Desktop/LVQU/fcn_init_weight.pkl','rb'))
#set_all_param_values(network['coarseflow'],params_fcn)
#print "FCN weight init done..."

#%%

num_epoches = 300
print "Number of Epoch: ",num_epoches
lr_decay_point = []

print "# ----"*9
print "starting training ..."
epoch_ssim=np.zeros((num_epoches,batch_size,3))
ite_train_hs,ite_vd_hs=[],[]
epoch_train,epoch_vd,epoch_test =[],[],[]

epoch_vd_mse_crs,epoch_vd_mse_rf=[],[]
epoch_test_mse_crs,epoch_test_mse_rf=[],[]

for epoch in range(num_epoches):
    # cross validation

    
    bt_train_hs,bt_vd_hs=[],[]
    bt_mse_crs,bt_mse_rf = [],[]
    

    batch1 = batch_generator( train_img[0:2048], batchsize = batch_size, shuffle = False )
    batch1 = progress( batch1, desc = 'Train Epoch %d/%d, Batch ' % (epoch , num_epoches),total = 2048//batch_size )
    for tbt in batch1:         
        train_loss = train_fn( tbt[:,0:1,:,:], tbt[:,1:2,:,:])
        bt_train_hs+=[train_loss]
        ite_train_hs+=[train_loss]
        
    batch2 = batch_generator( train_img[2048:2048+256], batchsize = batch_size, shuffle = False )
    batch2 = progress( batch2, desc = 'Validation Epoch %d/%d, Batch ' % (epoch , num_epoches),total = 256//batch_size )
    for vbt in batch2:
        vd_loss,vd_warp_rf,vd_warp_crs = vd_fn( vbt[:,0:1,:,:], vbt[:,1:2,:,:] )   
        ite_vd_hs+=[vd_loss]
        bt_vd_hs+=[vd_loss]
        bt_mse_crs+=[np.mean(np.abs(vd_warp_crs - vbt[:,1:2,:,:]))]
        bt_mse_rf+=[np.mean(np.abs(vd_warp_rf - vbt[:,1:2,:,:]))]

    epoch_train+=[np.mean(bt_train_hs)]
    epoch_vd+=[np.mean(bt_vd_hs)]
    epoch_vd_mse_crs+=[np.mean(bt_mse_crs)]
    epoch_vd_mse_rf+=[np.mean(bt_mse_rf)]
    print "Training Loss:",np.mean(bt_train_hs),", Vd Loss:",np.mean(bt_vd_hs),", MSE cosrse:",np.mean(bt_mse_crs),", MSE rf",np.mean(bt_mse_rf)



    for bt in batch_generator( test_img[0:batch_size], batchsize = batch_size, shuffle = False ):
        vd_loss,test_jsm,vd_warp_rf,tuv_rf,vd_warp_crs,tuv_crs,tuv2,tuv4,tuv8 = eval_fn( bt[:,0:1,:,:], bt[:,1:2,:,:] )   
    epoch_test+=[vd_loss]
    epoch_test_mse_crs+=[np.mean(np.abs(vd_warp_crs - bt[:,1:2,:,:]))]
    epoch_test_mse_rf+=[np.mean(np.abs(vd_warp_rf - bt[:,1:2,:,:]))]

    os.mkdir('result/'+str(epoch))
    for kk in range(batch_size):
        os.mkdir('result/'+str(epoch)+'/'+str(kk))

        imwrite('result/'+str(epoch)+'/'+str(kk)+'/im1.jpg',bt[kk,0].astype('uint8'))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/im2.jpg',bt[kk,1].astype('uint8'))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/coarse_warp.jpg',vd_warp_crs[kk,0].astype('uint8'))
        imwrite('result/'+str(epoch)+'/'+str(kk)+'/refine_warp.jpg',vd_warp_rf[kk,0].astype('uint8'))
        
        np.save('result/'+str(epoch)+'/'+str(kk)+'/jsm',test_jsm[kk,0])
        np.save('result/'+str(epoch)+'/'+str(kk)+'/refine_flow', tuv_rf[kk] )
        np.save('result/'+str(epoch)+'/'+str(kk)+'/coarse_flow', tuv_crs[kk] )
        np.save('result/'+str(epoch)+'/'+str(kk)+'/fcn_flow2', tuv2[kk] )
        np.save('result/'+str(epoch)+'/'+str(kk)+'/fcn_flow4', tuv4[kk] )
        np.save('result/'+str(epoch)+'/'+str(kk)+'/fcn_flow8', tuv8[kk] )
            
        epoch_ssim[epoch,kk,0]=ssim(bt[kk,0],bt[kk,1])
        epoch_ssim[epoch,kk,1]=ssim(bt[kk,1],vd_warp_crs[kk,0].astype('float64'))
        epoch_ssim[epoch,kk,2]=ssim(bt[kk,1],vd_warp_rf[kk,0].astype('float64'))       


# after training,save model.
values = lasagne.layers.get_all_param_values(network['refineflow'])
now = datetime.now()
pickle.dump( values, open( 'result/fcn_jsm_rf3'+now.strftime('%Y_%m_%d_%H_%M_%S')+'_model.pkl', 'w'))

f=plt.figure(figsize=(8,8))
plt.plot(epoch_train,'r',label='train')
plt.plot( epoch_vd,'b',label='validation')
plt.plot( epoch_test,'g',label='test')
plt.legend()
plt.grid('on')    
fn = 'result/EpochHS.jpg'
f.savefig(fn)
plt.close(f)

f=plt.figure(figsize=(8,8))
plt.plot(epoch_vd_mse_crs,'r',label='coarse')
plt.plot(epoch_vd_mse_rf,'b',label='refine')
plt.legend()
plt.grid('on')    
fn = 'result/EpochMSE_train.jpg'
f.savefig(fn)
plt.close(f)

f=plt.figure(figsize=(8,8))
plt.plot(epoch_test_mse_crs,'r',label='coarse')
plt.plot(epoch_test_mse_rf,'b',label='refine')
plt.grid('on')    
plt.legend()
fn = 'result/EpochMSE_test.jpg'
f.savefig(fn)
plt.close(f)

#save log
np.save('result/ite_train_hs',ite_train_hs)   
np.save('result/ite_vd_hs',ite_vd_hs)   

np.save('result/epoch_train',epoch_train)   
np.save('result/epoch_vd',epoch_vd)   
np.save('result/epoch_test',epoch_test) 

np.save('result/epoch_vd_mse_crs',epoch_vd_mse_crs )   
np.save('result/epoch_vd_mse_rf',epoch_vd_mse_rf)

np.save('result/epoch_test_mse_crs',epoch_test_mse_crs )   
np.save('result/epoch_test_mse_rf',epoch_test_mse_rf)
np.save('result/epoch_ssim',epoch_ssim)
print "stage_log saved."


