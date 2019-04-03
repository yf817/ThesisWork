#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import random
import time
import sys

from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import SimilarityTransform




from dask import delayed,threaded,compute
# --------------------------------------------------------------------------------
# version 3
# Supervised version batch generator, using module Dask + H5PY
# Dask: multi-thread to call h5py.__getitem__() function
# load shuffled batch on the fly in a acceptable time
# uint8 image : 0.18s,batchsize = 8
# float flow  : 0.25 s,batchsize = 8
# --------------------------------------------------------------------------------
# Supervised version batch generator
def get_batch1( h5image, h5flow, keys, batchsize, shuffle = True):
    indices = keys
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(keys) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        btimage = compute( [ delayed( h5image.__getitem__)(i) for i in excerpt], get = threaded.get)
        btflow  = compute( [ delayed( h5flow.__getitem__)(j) for j in excerpt], get = threaded.get)    
        yield np.stack(btimage)[0],np.stack(btflow)[0]  # [1,batchsize,ch,m,n]

# --------------------------------------------------------------------------------
# Unsupervised version batch generator
def get_batch2( h5image, keys, batchsize, shuffle = True):
    indices = keys
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(keys) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        btimage = compute( [ delayed(h5image.__getitem__)(i) for i in excerpt], get = threaded.get)
        yield np.stack(btimage)[0]
        
# --------------------------------------------------------------------------------
# Supervised Fetch specfic for testing
def fetch_batch1( h5image, h5flow, keys, batchsize ):
    for start_idx in range(0, len(keys) - batchsize + 1, batchsize ):
        excerpt = keys[start_idx:start_idx + batchsize]
        btimage = compute( [ delayed( h5image.__getitem__)(i) for i in excerpt], get = threaded.get)
        btflow  = compute( [ delayed( h5flow.__getitem__)(j) for j in excerpt], get = threaded.get)    
        yield np.stack(btimage)[0],np.stack(btflow)[0],excerpt   

# --------------------------------------------------------------------------------
# unSupervised Fetch specfic for testing
def fetch_batch2( h5image, keys, batchsize ):
    for start_idx in range(0, len(keys) - batchsize + 1, batchsize ):
        excerpt = keys[start_idx:start_idx + batchsize]
        btimage = compute( [ delayed(h5image.__getitem__)(i) for i in excerpt], get = threaded.get)
        yield np.stack(btimage)[0],excerpt   
        
        
# --------------------------------------------------------------------------------
# Monitoring batch generation
def progress(items, desc = '', total = None, min_delay = 0.0):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. 
    `total` gives the total number of items 
    `min_delay` gives the minimum time in seconds between subsequent prints. 
    `desc` gives an optional prefix text (end with a space).
    
    """
    
    total = total
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % ( desc, n+1, total, n / float(total) * 100), end=" ")
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print("(ETA: %d:%02d)" % divmod(t_total - t_done, 60), end=" ")
            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) + divmod(t_total, 60)))






# --------------------------------------------------------------------------------
# numpy file generator
def batch_generator( inputs, batchsize = 8, shuffle = False ):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)   
        yield inputs[excerpt,...]
        
def batch_generator_validation( data, batchsize = 8, cropshape=(256,512)):
    # for EPE Wwith GT
    for start_idx in range(0, len(data) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)   
        tmp = data[excerpt]
        o_bs,o_ch,o_m,o_n = tmp.shape
        
        crop_img = np.zeros((o_bs,6,cropshape[0],cropshape[1]))
        crop_flow = np.zeros((o_bs,2, cropshape[0],cropshape[1]))
        lx = random.sample(range(0, o_m-cropshape[0]),o_bs)
        ly = random.sample(range(0, o_n-cropshape[1]),o_bs)       
        for i,(x,y) in enumerate(zip(lx,ly)):
            crop_img[i,...] = tmp[i,0:6,x:x+cropshape[0],y:y+cropshape[1]]
            crop_flow[i,...] = tmp[i,6:8,x:x+cropshape[0],y:y+cropshape[1]]
        yield crop_img,crop_flow

# multi-thread speed up with queue
def thread_generator( generator, num_cached = 512 ):
    import Queue
    queue = Queue.Queue( maxsize = num_cached)
    sentinel = object()  
    
    def producer():
        for item in generator:
            queue.put(item)              
        queue.put(sentinel)

    import threading
    thread = threading.Thread( target = producer)
    thread.daemon = True
    thread.start()

    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()   
        
        
# --------------------------------------------------------------------------------
# data augmentation
def mean_subtraction_generator(generator):
    for item in generator:
        item -= np.mean(item, axis = 0)
        yield item
        
def normlization_generator(generator):
    for item in generator:
        item  /= np.std(item, axis = 0)
        yield item    
    
def random_crop_generator( generator,cropshape ):
    for item in generator:       
        o_bs,o_ch,o_m,o_n = item.shape
        crops = np.zeros((o_bs,o_ch,cropshape[0],cropshape[1]))
        lx = random.sample(range(0, o_m-cropshape[0]),o_bs)
        ly = random.sample(range(0, o_n-cropshape[1]),o_bs)       
        for i,(x,y) in enumerate(zip(lx,ly)):
            crops[i,...] = item[i,:,x:x+cropshape[0],y:y+cropshape[1]]          
        yield crops
            
def center_crop_generator(generator, center_crop):
    for data in generator:
        center = np.array(data.shape[2:])/2
        yield data[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)]

def randomfliplr_generator(generator, p = 0.5):
    for item in generator:
        bs = item.shape[0]
        l = np.sort(random.sample(range(0, bs),np.int( p*bs ) ))
        for i in l:
            item[i,...] =  item[i,:,:,::-1] 
        yield item        

def randomflipud_generator(generator, p = 0.5):
    for item in generator:
        bs = item.shape[0]
        l = np.sort(random.sample(range(0, bs),np.int( p*bs ) ))
        for i in l:
            item[i,...] =  item[i,:,::-1,:] 
        yield item

def rotation_generator(generator, angle = 17):
    for item in generator:
        bs,ch = item.shape[0],item.shape[1]
        agl = random.sample(range(-angle, angle),bs )
        for i,ag in enumerate(agl):
            for j in range(ch):
                item[i,j,...]= rotate( item[i,j,...], ag , preserve_range=True)
        yield item

def translate_generator(generator, p =(-0.2,0.2)):
    # p: translate ratio
    for batch in generator:
        bs,ch,h,w = batch.shape
        dx = random.sample(range( int(p[0]*h), int(p[1]*h) ), bs )
        dy = random.sample(range( int(p[0]*w), int(p[1]*w) ), bs )
        for i,tsl in enumerate(zip(dx,dy)):
            for j in range(ch):
                batch[i,j,...] = warp( batch[i,j,...], SimilarityTransform(translation = tsl), preserve_range=True)
        yield batch    
    
   
    
def aug_training_generator( data,batchsize,flage_Rcrop=True,flag_Msubtraction=False,flag_Norm=False, flag_RFlr=False,flag_RFup=False,flag_Rot=False,flag_Translate=False ):   
    generator = batch_generator(data,batchsize,shuffle=True)    
    if flage_Rcrop == True :
        generator = random_crop_generator(generator,cropshape=(256,512))
        
#    if flag_Msubtraction == True:
#        generator = mean_subtraction_generator(generator)
#        
#    if flag_Norm == True:
#        generator = normlization_generator(generator)
#        
#    if flag_RFlr == True:
#        generator = randomfliplr_generator(generator)
#        
#    if flag_RFup == True:
#        generator = randomflipud_generator(generator)    
#        
#    if flag_Rot == True:
#        generator = rotation_generator(generator)
#        
#    if flag_Translate == True:
#        generator = translate_generator(generator)
        
    for item in generator:
        yield item
             




# --------------------------------------------------------------------------------
#  Data Dugmentation in background thread
# only for training process
def augument_batch_in_background( generator , num_cached = 100 ):
    import Queue
    queue = Queue.Queue( maxsize = num_cached)
    sentinel = object()  # guaranteed unique reference


    # -------------------------------------------------------------------------
    # geometrical operaters
    # --- random crop ---
    def op_crop( batch, k_num = 5, cropshape=(300,500) ):   
        bs,_,m,n = batch.shape   
        if(cropshape[0] >= m-1 or cropshape[1]>= n-1):
            raise ValueError("Crop size too big.")
        else:
            dx = m-cropshape[0]
            dy = n-cropshape[1]
            lx = random.sample(range(0, dx),bs)
            ly = random.sample(range(0, dy),bs)       
            for i,(x,y) in enumerate(zip(lx,ly)):
                batch[i,...] = batch[i,:,x:x+cropshape[0],y:y+cropshape[1]]
        return batch
        
    # --- flip left to right ---                      
    def op_fliplr( batch , p = 0.5 ): 
        bs,_,_,_ = batch.shape
        l = np.sort(random.sample(range(0, bs),np.int( p*bs ) ))
        for i in l:
            batch[i,...] =  batch[i,:,:,::-1] 
        return batch
            
    # --- flip up to bottom ---              
    def op_flipud( batch, p = 0.5 ): 
        bs,_,_,_ = batch.shape
        l = np.sort(random.sample(range(0, bs),np.int( p*bs ) ))
        for i in l:
            batch[i,...] = batch[i,:,::-1,:]
        return batch
       
    # --- rotation ---
    def op_rot(batch , angle = (-17,17)):
        bs,ch,_,_ = batch.shape
        agl = random.sample(range(-17, 17),bs )
        for i,ag in enumerate(agl):
            for j in range(ch):
                batch[i,j,...]= rotate( batch[i,j,...], ag , preserve_range=True)
        return batch
    
    # --- translation ---
    def op_translate( batch , p = (-0.2,0.2)):
        bs,ch,h,w = batch.shape
        dx = random.sample(range( int(p[0]*h), int(p[1]*h) ), bs )
        dy = random.sample(range( int(p[0]*w), int(p[1]*w) ), bs )
        for i,tsl in enumerate(zip(dx,dy)):
            for j in range(ch):
                batch[i,j,...] = warp( batch[i,j,...], SimilarityTransform(translation = tsl), preserve_range=True)
        return batch

    # -------------------------------------------------------------------------
    # define producer 
    # generating and putting augumented items into queue
    def producer():
        for item in generator:
            if random.choice([True, False]):
                queue.put(item)
                
            # flip/rot/translation randomly put
            if random.choice([True, False]) :    
                item = op_fliplr(item)
                queue.put(item)
                
            if random.choice([True, False]):
                item = op_fliplr(item)
                queue.put(item)
                
            if random.choice([True, False]):
                item = op_rot(item)
                queue.put(item)
                
            if random.choice([True, False]):
                item = op_translate(item)
                queue.put(item)
                
            # cropping within one batch
#            if random.choice([True, False]):
#                item = op_crop(item)
#                queue.put(item)
                
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread( target = producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()





# --------------------------------------------------------------------------------
# batch generater
# version 1 : LMDB + Cpickle/Json  | slow/large  deleted
# version 2 : H5PY + fancy indexing   | suffle indexing not support & load into mem  flychairs:64G
# version 2 : H5PY + Dask + __getitem__()  | fast/small , multi-thread reading & on the fly loading


## version 2
## working directly with numpy array 
## drawback: load into workspace, not enough MEM; you have to cut dataset into pieces
## --------------------------------------------------------------------------------
## Supervised version batch generator
#def iterate_minibatches(image1, image2, flow, batchsize, shuffle = False):
#    """
#    Generates one epoch of batches of inputs and targets, optionally shuffled.
#    """
#    assert len(image1) == len(image2)
#    assert len(image1) == len(flow)
#    indices = range(len(flow))
#    if shuffle:
#        np.random.shuffle(indices)
#    for start_idx in range(0, len(image1) - batchsize + 1, batchsize):
#        excerpt = indices[start_idx:start_idx + batchsize]
#        yield image1[excerpt],image2[excerpt],flow[excerpt]
#        
#
#
## --------------------------------------------------------------------------------
## Supervised version batch generator
#def iterate_minibatches1(image1, image2, flow, keys, batchsize, shuffle = False):
#    """
#    Generates one epoch of batches of inputs and targets, optionally shuffled.
#    """
#    assert len(image1) == len(image2)
#    assert len(image1) == len(flow)
#    indices = keys
#    if shuffle:
#        np.random.shuffle(indices)
#        
#    for start_idx in range(0, len(image1) - batchsize + 1, batchsize):
#        excerpt = indices[start_idx:start_idx + batchsize]
#             
#        tmp1,tmp2,tmp3 = [],[],[]
#        for id in excerpt:
#            # reading slice
#            tmp1 += [ image1[id] ]
#            tmp2 += [ image2[id] ]
#            tmp3 += [ flow[id] ]
#        t1,t2,t3 = np.stack(tmp1),np.stack(tmp2),np.stack(tmp3)
#       
#        yield t1,t2,t3
#
#
#
## --------------------------------------------------------------------------------
## Unsupervised version batch generator
#def iterate_minibatches2(image1, image2, keys,batchsize, shuffle = False):
#    """
#    Generates one epoch of batches of inputs and targets, optionally shuffled.
#    
#    """
#    assert len(image1) == len(image2)
#    indices =  keys
#    if shuffle:
#        np.random.shuffle(indices)
#    for start_idx in range(0, len(image1) - batchsize + 1, batchsize):
#        if shuffle:
#            excerpt = indices[start_idx:start_idx + batchsize]
#        else:
#            excerpt = range(start_idx, start_idx + batchsize)
#         
#        tmp1,tmp2 = [],[]
#        for id in excerpt:
#            tmp1 += [ image1[id] ]
#            tmp2 += [ image2[id] ]
#        t1,t2 = np.stack(tmp1),np.stack(tmp2)
#           
#        yield  t1,t2
## --------------------------------------------------------------------------------
## Background queue thead to reducing batch waiting
## speed up I/O performance a lot  
#def background_generator(generator, num_cached = 10 ):
#    """
#    Runs a generator in a background thread, caching up to `num_cached` items.
#    """
#    import Queue
#    queue = Queue.Queue( maxsize = num_cached )
#    sentinel = object()  # guaranteed unique reference
#
#    # define producer (putting items into queue)
#    def producer():
#        for item in generator:
#            queue.put(item)
#        queue.put(sentinel)
#
#    # start producer (in a background thread)
#    import threading
#    thread = threading.Thread( target = producer )
#    thread.daemon = True
#    thread.start()
#    
#    # run as consumer (read items from queue, in current thread)
#    item = queue.get()
#    while item is not sentinel:
#        yield item
#        item = queue.get()
#            
        



