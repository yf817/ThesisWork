#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import theano.tensor as T
import theano
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



