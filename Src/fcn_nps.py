#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import os
import sys
import logging
from near_ps import Near_Ps
from numpy import *
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Fcn_NPS(object):
    """docstring for Fcn_NPS."""
    def __init__(self, img_folder,sets_file="./Datasets/sets.json"):
        super(Fcn_NPS, self).__init__()
        self.sets_file = os.path.abspath(sets_file)
        self.img_folder = os.path.abspath(img_folder)
        self.nimgs = 0
        self.nrows = 0
        self.ncols = 0
        self.nchannels = 0
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("near_ps")
        self.logger.setLevel(logging.DEBUG)

        self.load_sets()
        self.load_params()
        self.load_imgs()
        print "start"
        nps = Near_Ps(self.data,self.calib,self.params)
        XYZ = nps.main_fcn()
        print XYZ

    def load_sets(self):
        f = open(self.sets_file,'r')
        sets = json.load(f)
        S = array(sets['S'])
        K = array(sets['K'])
        Phi = array(sets['Phi'])
        Dir = array(sets['Dir'])
        mu = array(sets['mu'])
        f.close()

        self.calib = {'S':S, 'Dir':Dir, 'mu':mu, 'Phi':Phi, 'K':K}
        self.nimgs = S.shape[0]

    def load_params(self):
        self.params = {}
        self.params['z0'] = 700
        self.params['maxit'] = 100
        self.params['estimator'] = 'LS'
        self.params['indices'] = arange(1,self.nimgs-2)
        self.params['ratio'] = 300
        self.params['self_shadows'] = 0
        self.params['display'] = 0

    def load_imgs(self):
        Iamb = mpimg.imread(self.img_folder+'/photometric_sample_raw_ambient.png')
        if Iamb is None:
            self.logger.error("can not find file 'photometric_sample_raw_ambient.png' in folder"+self.img_folder)
        mask = mpimg.imread(self.img_folder+'/photometric_sample_mask_raw.png')
        if mask is None:
            self.logger.error("can not find file 'photometric_sample_mask_raw.png' in folder"+self.img_folder)
        self.nrows = Iamb.shape[0]
        self.ncols = Iamb.shape[1]
        if (len(Iamb.shape)>2):
            self.nchannels = Iamb.shape[2]
        else :
            self.nchannels = 1

        [xx,yy] = meshgrid(arange(1,self.ncols+1),arange(1,self.nrows+1))
        xx = xx - self.calib['K'][0,2]
        yy = yy - self.calib['K'][1,2]
        ff = (self.calib['K'][0,0]+self.calib['K'][1,1])/2
        cos4a = pow((ff/sqrt(xx*xx+yy*yy+ff*ff)),4)
        I = zeros((self.nimgs,self.nchannels,self.nrows,self.ncols))

        for i in range(0,self.nimgs):
            Ii = mpimg.imread(self.img_folder+'/photometric_sample_raw_%04d.png'%(i+1))
            for ch in range(self.nchannels):
                Ii[:,:,ch] = (Ii[:,:,ch]-Iamb[:,:,ch])/cos4a
                I[i,ch] = Ii[:,:,ch]
        I[I<0]=0

        self.nchannels = 1
        I = mean(I,1).reshape(self.nimgs,self.nchannels,self.nrows,self.ncols)
        mask = mean(mask,2)

        self.data = {}
        self.data['I'] = I
        self.data['mask'] = mask
        self.data['nrows'] = self.nrows
        self.data['ncols'] = self.ncols
        self.data['nchannels'] = self.nchannels
        self.data['nimgs'] = self.nimgs

if __name__ == "__main__":
    a = Fcn_NPS("./Example_Img/Buddha")
