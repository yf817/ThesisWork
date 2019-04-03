from theano.tensor.nnet import conv2d  as conv
import theano
import theano.tensor as T
from astropy.convolution import Gaussian2DKernel
import numpy as np



def _grad( img ): 
    bc,ch,h,w = img.shape
    # color image
#    im = img[:,0:1,:,:]*0.299 + img[:,1:2,:,:]*0.587 + img[:,2:3,:,:]*0.114
 
    im=img 

    ph = im[:,:,:-1,:]- im[:,:,1:,:]
    pw = im[:,:,:,:-1]- im[:,:,:,1:]   
    gx = T.zeros((bc,1,h,w))
    gy = T.zeros((bc,1,h,w))
    gx = T.set_subtensor( gx[:,:,:-1,:] ,ph )
    gy = T.set_subtensor( gy[:,:,:,:-1] ,pw )
   
    return gx,gy



def _smooth_t( t ):
    k = np.zeros((1,1,13,13))
    k[:,:,...]= Gaussian2DKernel(1.5).array 
    kk =  T.cast(k, theano.config.floatX) 
    st = conv( t, kk, border_mode = (6,6) )
    return st

def _smooth_tensor( image ):
    gx,gy  = _grad( image ) 
    t_xx = _smooth_t(gx**2)
    t_xy = _smooth_t(gx*gy)
    t_yx = _smooth_t(gx*gy)
    t_yy = _smooth_t(gy**2)
    ST = T.concatenate((t_xx,t_xy,t_yx,t_yy), axis = 1 )
    return ST


# tensor distance function
def _distance( va, vb ):
    coef = T.cast( 8*np.pi/15, theano.config.floatX) 
    dt = va-vb
    dtr = (dt[:,0] + dt[:,3])**2
    dtc = dt[:,0]**2 + dt[:,3]**2 + 2*dt[:,1]*dt[:,2]
    tmp = coef*( dtc - dtr/3.0 )
    ss = T.sqrt( T.maximum(tmp,0) )
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
def joint_saliancy( image1, image2 ):
    
    st_a = _smooth_tensor(image1)
    st_b = _smooth_tensor(image2)   
    s_a = _saliency(st_a)
    s_b = _saliency(st_b)    
    s_ab = _distance( st_a, st_b ).dimshuffle(0,'x',1,2)
    s_a  = _normalise(s_a)
    s_b  = _normalise(s_b)
    s_ab = _normalise(s_ab)

    j_s = T.minimum( s_a, s_b )*T.inv(s_ab + 1.0)  
    return s_a,s_b,j_s



#    st_a = _smooth_tensor(image1)
#    st_b = _smooth_tensor(image2)   
#    s_a =  _saliency(st_a)
#    s_b =  _saliency(st_b)
#    d_ab = _distance( st_a, st_b ).dimshuffle(0,'x',1,2)
#    s_ab = _normalise(d_ab)
#    jsm = _normalise(T.minimum( s_a, s_b ) )*T.inv( s_ab + 1.0)  
#    return jsm



#%%
#i1=imread('/home/fed/Desktop/oldimages/doll/fixed.jpg')
#i2=imread('/home/fed/Desktop/oldimages/doll/moving.jpg')
#im1=i1[:,:,0:1].transpose((2,0,1))
#im2=i2[:,:,0:1].transpose((2,0,1))
#im1=i1[np.newaxis,np.newaxis,...]
#im2=i2[np.newaxis,np.newaxis,...]
#
#res=ff(im1,im2)
#np.save('/home/fed/Desktop/oldimages/js_brain_f',res[0][0,0])
#np.save('/home/fed/Desktop/oldimages/js_brain_m',res[1][0,0])
#np.save('/home/fed/Desktop/oldimages/js_brain',res[2][0,0])
