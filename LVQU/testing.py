# -*- coding: utf-8 -*-
import os,shutil
wdir = r'/home/bme/Desktop/LVQU/3fcn_jsm_rf/'
os.chdir(wdir)

if not os.path.isdir('testing/'):
    os.mkdir( 'testing/')
else:
    shutil.rmtree('testing/')
    os.mkdir( 'testing/')
    

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
batch_size = 1
print "batch size is:\t",batch_size

st=time.time()
network = build_model(input_image1,input_image2,batch_size)
sp=time.time()
print "model building done, takes ",sp-st," s..."


vd_jsm,vd_rf,vd_crs,vd2,vd4,vd8 = get_output([network['jsm'],network['refineflow'],network['coarseflow'],network['Convolution3'],network['Convolution2'],network['Convolution1']],
        inputs = {  network['im1']:input_image1,network['im2']:input_image2 },
        deterministic = True ) 

vd_warp_rf = inverse_wraping(vd_rf, input_image1)
vd_warp_crs = inverse_wraping(vd_crs, input_image1)

#vd_hs_loss = HS_loss1(input_image2,vd_warp_rf,vd_rf)

st=time.time()
eval_fn = theano.function(
        [ input_image1, input_image2 ],
        [vd_jsm,vd_warp_rf,vd_rf,vd_warp_crs,vd_crs,vd2,vd4,vd8],
        allow_input_downcast = True )
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


test_img = np.load('/home/bme/Desktop/LVQU/data/LargeDisplacement_test_imagepair128.npy')


params= pickle.load(open('/home/bme/Desktop/LVQU/3fcn_jsm_rf/result/fcn_jsm_rf32018_11_16_08_14_20_model.pkl','rb'))
set_all_param_values(network['refineflow'],params)
print "loading data done..."

#%%
epoch=0
for bt in iterate_single(test_img,batchsize=1):
    print epoch
    test_jsm,warp_rf,uv_rf,warp_crs,uv_crs,tuv2,tuv4,tuv8 = eval_fn( bt[0][:,0:1,:,:], bt[1][:,0:1,:,:] )   
    os.mkdir('testing/'+str(epoch))
    imwrite('testing/'+str(epoch)+'/im1.jpg',bt[0][0,0].astype('uint8'))
    imwrite('testing/'+str(epoch)+'/im2.jpg',bt[1][0,0].astype('uint8'))
    imwrite('testing/'+str(epoch)+'/warp_rf.jpg',warp_rf[0,0].astype('uint8'))
    imwrite('testing/'+str(epoch)+'/warp_crs.jpg',warp_crs[0,0].astype('uint8'))
    
    np.save('testing/'+str(epoch)+'/jsm',test_jsm[0,0])
    np.save('testing/'+str(epoch)+'/refine_flow', uv_rf[0] )
    np.save('testing/'+str(epoch)+'/coarse_flow', uv_crs[0] )
    
    
    fig=plt.figure(figsize=(10,10))
    ax =fig.add_subplot(111)
    dr=1
    ax.quiver(uv_rf[0,0,::dr,::dr], -uv_rf[0,1,::dr,::dr],units='xy');
    plt.axis('off')
    plt.gca().invert_yaxis();ax.set_aspect(1)
    fig.savefig('testing/'+str(epoch)+'/flow_rf.jpg',bbox_inches='tight')
    plt.close(fig)   
    
    fig=plt.figure(figsize=(10,10))
    plt.imshow(test_jsm[0,0],cmap='jet')
    plt.axis('off')
    plt.subplots_adjust(hspace=0,wspace=0) 
    fig.savefig('testing/'+str(epoch)+'/jsm.jpg',bbox_inches='tight')
    plt.close(fig)   
    
    epoch = epoch +1






