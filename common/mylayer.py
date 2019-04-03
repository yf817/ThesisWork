#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# customise three layers based on theano

import theano
import theano.tensor as T
#from lasagne.utils import floatX
from lasagne.layers.base import Layer, MergeLayer
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
#from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv2d  as conv

from astropy.convolution import Gaussian2DKernel
import numpy as np


__all__ = [
    "WarpingLayer",  
    "JointSaliencyLayer",
    #"SaliencyLayer",
    "UpsamplingFlowLayer"
]



# -----------------------------------------------------------------------------
class UpsamplingFlowLayer(Layer):
    def get_output_shape_for(self, input_shape):
        return (self.input_shape[0],self.input_shape[1],self.input_shape[2]*2,self.input_shape[3]*2)
     
    def get_output_for(self, inputs, **kwargs):
        return bilinear_upsampling(inputs,ratio = 2)
    

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
class WarpingLayer(MergeLayer):
    def __init__(self, incoming_motion,moving_image ,**kwargs):
        
        super(WarpingLayer, self).__init__([incoming_motion,moving_image], **kwargs)
        motion_shape ,img_shape = self.input_shapes 

        if motion_shape[2:4] != img_shape[2:4]:
            raise ValueError("The motion and image must have same height and weight.")
        if motion_shape[1] != 2 :
            raise ValueError("The Motion shape should be:(batch,2,m,n) ")
    
    def get_output_shape_for(self, input_shapes):

        img_shape = input_shapes[1]
        return img_shape
    
    def get_output_for(self, inputs, **kwargs):
        '''
        Input: [motion: [batch,2,m,n], image: [batch,ch,m,n] ]  1:gray , 3:color 
        Output: Warped_Images: [batch,ch,m,n]
        '''
        motions,imgs = inputs
        results = inverse_wraping( motions, imgs )
        return results


def inverse_wraping( motions , imgs):
    
    num_batch, num_channels, height_, width_ = imgs.shape
    x_h,x_w = T.mgrid[0:height_, 0:width_]  #int64
    xp_h_flat = T.tile(T.reshape(x_h,(1,-1)),num_batch)  # [1,bs*height*width]
    xp_w_flat = T.tile(T.reshape(x_w,(1,-1)),num_batch)  # [1,bs*height*width]

    uv = motions.dimshuffle(0,2,3,1) 
    uv = T.reshape(uv,(-1,2))   # [bs*height*width,2]
    dx =  uv[...,0].dimshuffle('x',0)   # [1,bs*height*width]
    dy =  uv[...,1].dimshuffle('x',0)   # [1,bs*height*width]

    x_flat = T.cast(xp_w_flat,theano.config.floatX)- T.cast(dx,theano.config.floatX)   # x(float) = x'(int64) - dx(float32)
    y_flat = T.cast(xp_h_flat,theano.config.floatX)- T.cast(dy,theano.config.floatX)   # y(float) = y'(int64) - dy(float32)
    
    x_flat = x_flat.flatten()
    y_flat = y_flat.flatten()
    
    # dimshuffle back to  (bs, height, width, channels)
    input_dim = imgs.dimshuffle(0, 2, 3, 1)
    bilineared = _interpolate(input_dim, x_flat, y_flat )
    warp = T.reshape( bilineared, (num_batch, height_, width_, num_channels))
    output = warp.dimshuffle(0, 3, 1, 2)  
    return output


def _interpolate( input_ , warpgrid_x, warpgrid_y ):
    num_batch, height, width, channels = input_.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)
    x,y = warpgrid_x,warpgrid_y
    
    x0_f = T.floor( x )
    y0_f = T.floor( y )
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    
    # 1 clip out of boundary points 
    x0 = T.clip( x0_f, 0, width_f  - 1 )
    x1 = T.clip( x1_f, 0, width_f  - 1 )
    y0 = T.clip( y0_f, 0, height_f - 1 )
    y1 = T.clip( y1_f, 0, height_f - 1 )   
    x0, x1, y0, y1 = (T.cast(v, 'int64') for v in (x0, x1, y0, y1))
    
    
    # 2 convert to indexing in flatten vector
    dim2 = width
    dim1 = width*height
    base = T.repeat( T.arange(num_batch, dtype='int64')*dim1,  height*width )
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # 3 indexing and sum
    im_flat = T.reshape( input_, (-1, channels) )
    Ia = im_flat[ idx_a ]  #[num_batch*height*width, channels]
    Ib = im_flat[ idx_b ]
    Ic = im_flat[ idx_c ]
    Id = im_flat[ idx_d ]
    
    wa = T.repeat( ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x'),channels, axis= 1)
    wb = T.repeat( ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x'),channels, axis= 1)
    wc = T.repeat( ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x'),channels, axis= 1)
    wd = T.repeat( ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x'),channels, axis= 1)     
    res = wa*Ia + wb*Ib + wc*Ic + wd*Id    
    
    return res


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
class JointSaliencyLayer(MergeLayer):
    '''
    joint saliency extraction layer,
    input: [fix_image,warped_image] ,shape(b,ch,m,n)
    '''
    def __init__(self, fix_image,warped_image ,**kwargs):
        
        '''
        joint saliency into theano formulation
        '''
        super(JointSaliencyLayer, self).__init__([fix_image,warped_image], **kwargs)
        shape1,shape2 = self.input_shapes 
   
        if len(shape1) != 4 or len(shape2) != 4 :
            raise ValueError("Input should be 4d tensor")
        if shape1[1:4] != shape2[1:4]:
            raise ValueError("The fixed and moving image must have same [ch,height,width].")
       
            
    def get_output_shape_for(self, input_shapes):
        shape1,shape2 = input_shapes
        return (shape1[0],1,shape1[2],shape1[3])
        
    def get_output_for(self, inputs, **kwargs):
        f_im,w_im = inputs
        jsm = _JSM(f_im,w_im)
        return jsm


def _grad( img ): 
    bc,ch,h,w = img.shape
#    im = img[:,0:1,:,:]*0.299 + img[:,1:2,:,:]*0.587 + img[:,2:3,:,:]*0.114
    im=img 
    ph = im[:,:,:-1,:]- im[:,:,1:,:]
    pw = im[:,:,:,:-1]- im[:,:,:,1:]   
    gx = T.zeros((bc,1,h,w))
    gy = T.zeros((bc,1,h,w))
    gx = T.set_subtensor( gx[:,:,:-1,:] ,ph )
    gy = T.set_subtensor( gy[:,:,:,:-1] ,pw )
    edge = T.concatenate((gx,gy),axis=1)
    return edge

def _smooth_t( t ):
    k = np.zeros((1,1,9,9))
    k[:,:,...]= Gaussian2DKernel(1.0).array 
    kk =  T.cast(k, theano.config.floatX) 
    st = conv( t, kk, border_mode = (4,4) )
    return st

def _smooth_tensor( image ):
    edge  = _grad( image ) 
    t_xx = _smooth_t(edge[:,0:1,:,:]**2)
    t_xy = _smooth_t(edge[:,0:1,:,:]*edge[:,1:2,:,:])
    t_yx = _smooth_t(edge[:,0:1,:,:]*edge[:,1:2,:,:])
    t_yy = _smooth_t(edge[:,1:2,:,:]**2)
    ST = T.concatenate((t_xx,t_xy,t_yx,t_yy), axis = 1 )
    return ST


# tensor distance function
def _distance( va, vb ):
    coef = T.cast( 8*np.pi/15, theano.config.floatX) 
    dt = va-vb
    dtr = (dt[:,0] + dt[:,3])**2
    dtc = dt[:,0]**2 + dt[:,3]**2 + 2*dt[:,1]*dt[:,2]
    tmp = coef*( dtc - dtr/3.0 )
    ss = T.sqrt( T.maximum(tmp,0) + T.cast(1e-9,theano.config.floatX) )
    return ss


def _saliency( st ): 
    bs,k,m,n = st.shape
    # 8 neighbours' indexing & clip into valid region
    #  
    #   n1  n2  m3
    #   n4  x   n5
    #   n6  n7  n8  
    #
    idx,idy = T.mgrid[0:m,0:n]
    idx1,idy1 = T.clip(idx-1,0,m-1), T.clip(idy-1,0,n-1)
    idx2,idy2 = T.clip(idx-1,0,m-1), T.clip(idy  ,0,n-1)
    idx3,idy3 = T.clip(idx-1,0,m-1), T.clip(idy+1,0,n-1)
    idx4,idy4 = T.clip(idx  ,0,m-1), T.clip(idy-1,0,n-1)
    idx5,idy5 = T.clip(idx  ,0,m-1), T.clip(idy+1,0,n-1)
    idx6,idy6 = T.clip(idx+1,0,m-1), T.clip(idy-1,0,n-1)
    idx7,idy7 = T.clip(idx+1,0,m-1), T.clip(idy  ,0,n-1)
    idx8,idy8 = T.clip(idx+1,0,m-1), T.clip(idy+1,0,n-1)  

    # 8 neighbours' slicing
    n1 = st[:,:,idx1,idy1]
    n2 = st[:,:,idx2,idy2]
    n3 = st[:,:,idx3,idy3]
    n4 = st[:,:,idx4,idy4]
    n5 = st[:,:,idx5,idy5]
    n6 = st[:,:,idx6,idy6]
    n7 = st[:,:,idx7,idy7]
    n8 = st[:,:,idx8,idy8] 
    
    d1 = _distance( st, n1 )
    d2 = _distance( st, n2 )
    d3 = _distance( st, n3 )
    d4 = _distance( st, n4 )
    d5 = _distance( st, n5 )
    d6 = _distance( st, n6 )
    d7 = _distance( st, n7 )
    d8 = _distance( st, n8 )
       
    ss = ( d1+d2+d3+d4+d5+d6+d7+d8 ) / 9.0
    distance = ss.dimshuffle(0,'x',1,2)
    return distance

def _normalise(tt):
    ch_max = T.max(tt,axis=(0,2,3),keepdims=True)
    ch_min = T.min(tt,axis=(0,2,3),keepdims=True)
    tmp = (tt-ch_min)/ch_max 
    return tmp


# joint saliency
def _JSM( image1, image2 ):
    st_a = _smooth_tensor(image1)
    st_b = _smooth_tensor(image2)   
    s_a =  _saliency(st_a)
    s_b =  _saliency(st_b)
    d_ab = _distance( st_a, st_b ).dimshuffle(0,'x',1,2)
    s_ab = _normalise(d_ab)
    jsm = _normalise( T.minimum( s_a, s_b )*T.inv( s_ab + 1.0) )
    return jsm


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
'''
class SaliencyLayer(Layer):
    
    #single saliency extraction layer, ch = 1
    
           
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0],1,input_shapes[2],input_shapes[3])
        
    def get_output_for(self, inputs, **kwargs):
        f_im = inputs
        return Saliency(f_im)


def Saliency( image ):   
    bs,ch,m,n = image.shape
    st = _smooth_tensor( image )
    
    # 8 neighbours' indexing & clip into valid region
    #  
    #   n1  n2  m3
    #   n4  x   n5
    #   n6  n7  n8  
    #
    idx,idy = T.mgrid[0:m,0:n]
    idx1,idy1 = T.clip(idx-1,0,m-1), T.clip(idy-1,0,n-1)
    idx2,idy2 = T.clip(idx-1,0,m-1), T.clip(idy  ,0,n-1)
    idx3,idy3 = T.clip(idx-1,0,m-1), T.clip(idy+1,0,n-1)
    idx4,idy4 = T.clip(idx  ,0,m-1), T.clip(idy-1,0,n-1)
    idx5,idy5 = T.clip(idx  ,0,m-1), T.clip(idy+1,0,n-1)
    idx6,idy6 = T.clip(idx+1,0,m-1), T.clip(idy-1,0,n-1)
    idx7,idy7 = T.clip(idx+1,0,m-1), T.clip(idy  ,0,n-1)
    idx8,idy8 = T.clip(idx+1,0,m-1), T.clip(idy+1,0,n-1)  
    
    # slicing
    n1 = st[:,:,idx1,idy1]
    n2 = st[:,:,idx2,idy2]
    n3 = st[:,:,idx3,idy3]
    n4 = st[:,:,idx4,idy4]
    n5 = st[:,:,idx5,idy5]
    n6 = st[:,:,idx6,idy6]
    n7 = st[:,:,idx7,idy7]
    n8 = st[:,:,idx8,idy8] 
    
    d1 = _distance( st, n1 )
    d2 = _distance( st, n2 )
    d3 = _distance( st, n3 )
    d4 = _distance( st, n4 )
    d5 = _distance( st, n5 )
    d6 = _distance( st, n6 )
    d7 = _distance( st, n7 )
    d8 = _distance( st, n8 )
       
    ss = ( d1+d2+d3+d4+d5+d6+d7+d8 ) / 9.0
    distance = ss.dimshuffle(0,'x',1,2)
    return  _normalise(distance) 

'''
