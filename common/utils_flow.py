#!/usr/bin/python
"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""

import scipy,png
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from scipy.misc import imsave

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

# --------------------------------------------------------------------------------
# display helper function
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Compare Ground Truth and Prediction Flow batch 
# Supervised version
def show_flow_batch1( pd, bt ):
    '''
    pd: preddiction batch,
    bt: input batch: [image1,image2,GT_flow]
    '''
    im1,im2,gt,keys = bt[0][:,0:3,:,:],bt[0][:,3:6,:,:],bt[1],bt[2]
    batch,ch,m,n = gt.shape
    fn = [ '{0:05}'.format(id) for id in keys]
    t1,t2 = gt.transpose(0,2,3,1)   ,   pd.transpose(0,2,3,1)
    i1,i2 = im1.transpose(0,2,3,1)  ,   im2.transpose(0,2,3,1)
    
    for i in range(batch):
        plt.figure(figsize=(16,4))
        plt.subplot(141)
        plt.imshow(flow_to_image(t1[i])),plt.axis('off'),plt.title('GT_'+fn[i])      
        plt.subplot(142)
        plt.imshow(flow_to_image(t2[i])),plt.axis('off'),plt.title('Prediction_'+fn[i])
        plt.subplot(143)
        plt.imshow(i1[i]),plt.axis('off'),plt.title('Image1_'+ fn[i])
        plt.subplot(144)
        plt.imshow(i2[i]),plt.axis('off'),plt.title('Image2_'+fn[i])
        plt.show()


# --------------------------------------------------------------------------------
# Compare Ground Truth and Prediction Flow batch 
# UnSupervised version : without GT plot 
def show_flow_batch2( pd, bt ):
    '''
    pd: preddiction batch,
    bt: input batch: [image1,image2,GT_flow]
    '''
    im1,im2,keys = bt[0][:,0:3,:,] , bt[0][:,3:6,:,:] , bt[1]
    fn = [ '{0:05}'.format(id) for id in keys]
    batch,ch,m,n = im1.shape
    t2 =pd.transpose(0,2,3,1)
    i1,i2 = im1.transpose(0,2,3,1),im2.transpose(0,2,3,1)
    for i in range(batch):
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(flow_to_image(t2[i])),plt.axis('off'),plt.title('Prediction_'+fn[i])
        plt.subplot(132)
        plt.imshow(i1[i]),plt.axis('off'),plt.title('Image1_'+fn[i])
        plt.subplot(133)
        plt.imshow(i2[i]),plt.axis('off'),plt.title('Image2_'+fn[i])
        plt.show()    

# --------------------------------------------------------------------------------
# visualize each (height, width) thing in a grid of size approx. 
def arrange_tensor3( data ):
    """
    Convert 3d tendor to 2d padded matrix for visulisation
    Input: shape: (n, height, width) 
    Output : shape (sqrt(n) , sqrt(n))
    """
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),(0, 1), (0, 1))  + ((0, 0),) * (data.ndim - 3))  
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data


# --------------------------------------------------------------------------------
# visualise Convlutional layer kernel
def vis_conv_kernel( kernel ):
    '''
    kernel: get_value(layer)
    '''
    plt.imshow( arrange_tensor3(kernel),cmap='gray')
    plt.axis('off')
    plt.show()


# --------------------------------------------------------------------------------
# visualise Convlutional layer 4d-tensor Output 
def vis_layer_output( output ):
    '''
    output: get_output(net['layer']) 
    if prediction flow (batch,2,m,n), direclty displaying.
    '''
    b,ch,m,n = output.shape
    if len(output.shape) != 4:
        print "wrong weight format"
        return 1
    dd=[]
    if (ch != 2):
        # norm 4-d tensor, not prediction flow
        for item in output:
            tmp = arrange_tensor3( item )
            dd += [tmp]
        data = np.stack(dd)
        plt.imshow(arrange_tensor3(data),cmap='gray'), plt.axis('off')
        plt.show()
    else:
        # display prediction flow      
        tt = output.transpose(0,2,3,1)
        for i,item in enumerate( tt ):
            plt.figure(figsize=(4,4))
            plt.imshow(flow_to_image( item ))
            plt.axis('off')
            plt.title('Batch_'+str(i+1))
        


# --------------------------------------------------------------------------------
# saving to files helper
# --------------------------------------------------------------------------------
# Prediction Flow batch to flo.file
def save_flow2file( bt, keys,folder ):
    keys = [ '{0:05}'.format(id) for id in keys]
    for f,kk in zip(bt,keys):
        fn = folder+ kk +'_pd.flo'
        write_flow(f.transpose(1,2,0),fn)
        
# --------------------------------------------------------------------------------
# Prediction Flow batch to jpg
def save_flow2image( pd, keys,folder):
    keys = [ '{0:05}'.format(id) for id in keys]
    t = pd.transpose(0,2,3,1)
    for it,kk in zip(t,keys):
        tmp = flow_to_image(it)
        imsave(folder + kk +'_pd.jpg',tmp)
        
# --------------------------------------------------------------------------------
# GT Flow batch to jpg
def save_gtflow2image( gt, keys, folder ):
    keys = [ '{0:05}'.format(id) for id in keys]
    t = gt.transpose(0,2,3,1)
    for it,kk in zip(t,keys):
        tmp = flow_to_image(it)
        imsave(folder + kk + '_gt.jpg',tmp)
        
        



    
    
# --------------------------------------------------------------------------------
# Flow helper function   
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# https://github.com/liruoteng/OpticalFlowToolkit
# ==============================  
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.show()


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        #print "Reading %d x %d flo file" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d
    
    
def read_flow_png(flow_file):
    """
    Read optical flow from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow
    
def visualize_flow(flow, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow)
        plt.imshow(img)
        plt.show()
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        max_flow = max(np.max(du), np.max(dv))
        img = np.zeros((h, w, 3), dtype=np.float64)
        # angle layer
        img[:, :, 0] = np.arctan2(dv, du) / (2 * np.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = np.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid
        # show
        plt.imshow(img)
        plt.show()

    return None


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()



def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    if (flow.shape[0] == 2 ):
       # [2,m,n] -> [m,n,2]
       flow = flow.transpose((1,2,0))
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel    
    


def warp_image(im, flow):
    '''
    warping 2d image with flow [m,n,2]
    '''
    hh = im.shape[0]
    ww = im.shape[1]
    
    (idh, idw) = np.mgrid[0:hh, 0:ww]
    #inverse warping   ww:hor  hh:vect
    xwp = idw.astype('float64') - flow[:,:,0]
    xhp = idh.astype('float64') + flow[:,:,1]
    x0,y0 = np.floor(xhp),np.floor(xwp)
    ax,ay = xhp-x0,xwp-y0
    
    #bilinear interpolation 4 neighbours 
    iax,iay = np.clip(x0.astype('int64'),0,ww-1),   np.clip(y0.astype('int64'),0,hh-1)
    ibx,iby = np.clip(x0.astype('int64')+1,0,ww-1), np.clip(y0.astype('int64'),0,hh-1)
    icx,icy = np.clip(x0.astype('int64'),0,ww-1),   np.clip(y0.astype('int64')+1,0,hh-1)
    idx,idy = np.clip(x0.astype('int64')+1,0,ww-1), np.clip(y0.astype('int64')+1,0,hh-1)
    wp = im[iax,iay]*(1-ax)*(1-ay)+im[ibx,iby]*ax*(1-ay)+im[icx,icy]*(1-ax)*ay+im[idx,idy]*ax*ay
    return wp



def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mepe

