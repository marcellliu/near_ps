#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import os
import sys
from numpy import *
import near_ps

class Fcn_NPS(object):
    """docstring for Fcn_NPS."""
    def __init__(self, img_folder,sets_file="./Datasets/sets.json",imgs_file="./Datasets/light_sets.json"):
        super(Fcn_NPS, self).__init__()
        self.sets_file = os.path.abspath(sets_file)
        self.img_folder = os.path.abspath(img_folder)
        self.nimgs = 0;

        self.load_sets()
        self.load_params()
        self.load_imgs()

    def load_sets(self):
        f = open(self.sets_file,'r')
        sets = json.load(f)
        S = array(sets['S'])
        K = array(sets['K'])
        Phi = array(sets['Phi'])
        Dir = array(sets['Dir'])
        mu = array(sets['mu'])

        calib = {'S':S, 'Dir':Dir, 'mu':mu, 'Phi':Phi, 'K':K}
        self.nimgs = S.shape[0]
        return calib

    def load_params(self):
        params = {}
        params['z0'] = 700
        params['estimator'] = 'LS'
        params['indices'] = arange(1,self.nimgs-2)
        params['ratio'] = 10
        params['self_shadows'] = 0
        params['display'] = 0

        return params

    def load_imgs(self):
        print self.img_folder
        for filename in os.listdir(self.img_folder):
            print (filename)

if __name__ == "__main__":
    a = Fcn_NPS("./Example_Img/Buddha")
