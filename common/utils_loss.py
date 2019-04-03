#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import theano.tensor as T
from theano.tensor.nnet import conv2d

import numpy as np

# --------------------------------------------------------------------------------
# penalty function
def rpo(x):
    # eq to y=|x|
    return T.sqrt(x**2 + 0.001**2)

def grpo( x, alp = 0.45):
    return (x**2 + 0.0001**2)**alp



#---------------------------------------------------------------------------------
# efficient version
# direct matrix op

def FlowSmooth(uv):
    # without asignment op, sum directly
    xx = rpo( uv[:,:,:-1,:]- uv[:,:,1:,:])
    yy = rpo( uv[:,:,:,:-1]- uv[:,:,:,1:])
    res = T.sum(xx) + T.sum(yy)
    return res


def Image_GD(im):
    # first order image gradient
    ph = im[:,:,:-1,:]- im[:,:,1:,:]
    pw = im[:,:,:,:-1]- im[:,:,:,1:]
    bc,ch,h,w = im.shape 
    padx = T.zeros((bc,ch,h,w))
    pady = T.zeros((bc,ch,h,w))
    padx = T.set_subtensor( padx[:,:,:-1,:] ,ph )
    pady = T.set_subtensor( pady[:,:,:,:-1] ,pw )
    edge = T.sqrt( padx**2 + pady**2 + np.finfo(float).eps )
    return edge

def ImageGDD(im1,im2):
    edge1 = Image_GD(im1)
    edge2 = Image_GD(im2)
    return T.sum(rpo(edge1-edge2))

def ImageD(im_f,im_w):
    diff = T.sum(rpo(im_f - im_w ))
    return diff

# energy loss function
def HS_loss2( im_f, im_w, motion ):
    l_data = ImageD(im_f , im_w)
    l_smooth = FlowSmooth(motion)
    res = l_data  + 1*l_smooth
    return   res


def HS_loss1( im_f, im_w, motion ):
    l_data = ImageD(im_f , im_w) + 20*ImageGDD(im_f , im_w)
    l_smooth = FlowSmooth(motion)
    res = l_data + 1*l_smooth
    return   res



#def p_smooth_l2(x):
#    # first-order smooth
#    kx = np.zeros((2,2,1,3)) 
#    kx[:,:,...] = np.array([1,-1,0]).reshape((1,3)) 
#    tx = T.cast(kx, theano.config.floatX)
#    gdx = conv2d(x,tx,border_mode=(0,1))
#    
#    ky= np.zeros((2,2,3,1)) 
#    ky[:,:,...] = np.array([1,-1,0]).reshape((3,1))
#    ty = T.cast(ky, theano.config.floatX)
#    gdy = conv2d(x,ty,border_mode=(1,0))
#    
#    res = T.sum( T.sqrt(gdx**2 + gdy**2) ) 
#    return res
#def p_image_gradient_l2(im1,im2):
#    # image gradient with sobel kernel
#    sobelx =  np.array([[1,0,-1],[2,0,-2],[1,0,-1]])*0.25
#    sobely =  np.array([[1,2,1],[0,0,0],[-1,-2,-1]])*0.25
#    # color image
#    kx = np.zeros((1,3,3,3))
#    ky = np.zeros((1,3,3,3))
#    
#    kx[:,:,...] = sobelx
#    tx = T.cast(kx, theano.config.floatX)
#    ky[:,:,...] = sobely
#    ty = T.cast(ky, theano.config.floatX)
#    
#    im1_gx  = conv2d( im1, tx, border_mode = (1,1) )
#    im1_gy  = conv2d( im1, ty, border_mode = (1,1) )
#    im2_gx  = conv2d( im2, tx, border_mode = (1,1) )
#    im2_gy  = conv2d( im2, ty, border_mode = (1,1) )
#
#    im1_gd = T.sqrt( im1_gx**2 + im1_gy**2 ) 
#    im2_gd = T.sqrt( im2_gx**2 + im2_gy**2 )  
#    return T.sum( rpo(im1_gd - im2_gd) )




def EPE(pd,gt):
    res = T.mean( T.sqrt( (pd[:,0,:,:]-gt[:,0,:,:])**2+(pd[:,1,:,:]-gt[:,1,:,:])**2 +np.finfo(float).eps )  )
    return res

def array_EPE(pd,gt):
    tmp = T.sqrt( (pd[:,0,:,:]-gt[:,0,:,:])**2+(pd[:,1,:,:]-gt[:,1,:,:])**2  )
    return T.mean(tmp,(1,2))


#================================
def CC(I,J):
    I2 = I*I
    J2 = J*J
    IJ = I*J
    # patch windows size is 11,11
    sum_filter = T.ones((1,3,11,11))
    
    I_sum = conv2d(I,sum_filter,border_mode='half')
    J_sum = conv2d(J,sum_filter,border_mode='half')
    
    I2_sum = conv2d(I2,sum_filter,border_mode='half')
    J2_sum = conv2d(J2,sum_filter,border_mode='half')
    IJ_sum = conv2d(IJ,sum_filter,border_mode='half')
    
    win_size = 11.0*11.0
    I_mean = I_sum/win_size
    J_mean = J_sum/win_size
    
    cross = IJ_sum - I_sum*J_mean- I_mean*J_sum + I_mean*J_mean*win_size
    I_var = I2_sum - 2*I_mean*I_sum + I_mean*I_mean*win_size
    J_var = J2_sum - 2*J_mean*J_sum + J_mean*J_mean*win_size
    
    cc = cross*cross/( I_var*J_var + np.finfo(float).eps) 
    res = T.sum(T.clip(cc,0,1))
    
    return res

