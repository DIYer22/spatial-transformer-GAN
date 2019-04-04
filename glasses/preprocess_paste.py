#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DIYer22@github
@mail: ylxx@live.com
Created on Wed Apr  3 16:44:01 2019
"""
from boxx import *
from boxx import makedirs, np, npa, imread, pathjoin, glob, os, resize, p, uint8

ignoreWarning()


celebA_path = os.path.expanduser('~/dataset/celeba')
dataset = pathjoin(celebA_path, 'eyeglasses_stgan_dataset')

st_gan_dataset = pathjoin(dataset, 'tf_st_gan_dataset')

psa = sorted(glob(pathjoin(dataset, 'trainA/*')))[0::2]
psb = sorted(glob(pathjoin(dataset, 'trainB/*')))[1::2]

lena = len(psa)
lenb = len(psb)

attribute = np.zeros((lena+lenb, 40), np.bool)
attribute[-lenb:,15] = True

img = imread(psa[0])
shape = img.shape

makedirs(p/st_gan_dataset)
imgns = psa+psb

if shape[:2] != (218, 178,):
    def f(imgn):
        return  uint8(resize(imread(imgn),(218, 178,)))
else:
    def f(imgn):
        imread(imgn)

if __name__ == '__main__':  
    imgns = imgns[:]
#    with timeit():
#        images = mapmp(f, imgns, pool=16)
#    with timeit():
#        images = mapmp(f, imgns, pool=8)
#    with timeit():
#        images = mapmt(f, imgns, pool=16)
    with timeit():
        images = mapmt(f, imgns, pool=8)
#    with timeit():
#        images = map2(f, imgns, )
    images = np.array((images))


    attrp = pathjoin(st_gan_dataset, 'attribute_train.npy')
    imagep = p/pathjoin(st_gan_dataset, 'image_train.npy')
    
    np.save(attrp, attribute)
    np.save(imagep, images)


#    os.link(attrp, attrp.replace('train.npy', 'test.npy'))
#    os.link(imagep, imagep.replace('train.npy', 'test.npy'))

if __name__ == "__main__":
    pass
    
    
    
