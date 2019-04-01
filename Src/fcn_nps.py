# -*- coding:utf-8 -*-

import os
import sys
import logging
from ps import ps
import numpy as np
from scipy.io import loadmat
from PIL import Image
# import Image

class Fcn_NPS(object):
    """docstring for Fcn_NPS."""
    def __init__(self, tagetfolder,sets_file="./Datasets/sets.json"):
        super(Fcn_NPS, self).__init__()
        self.tagetfolder = os.path.abspath(tagetfolder)
        self.nimgs = 0
        self.nrows = 0
        self.ncols = 0
        self.nchannels = 0
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("near_ps")
        self.logger.setLevel(logging.DEBUG)

        self.loaddata()
        print('start')
        # nps = ps(self.data,self.calib,self.params)
        # XYZ,N = nps.main_fcn()
        # XYZ[0,:] = XYZ[0,:]-sum(XYZ[0,:])/XYZ.shape[2]
        # XYZ[1,:] = XYZ[1,:]-sum(XYZ[1,:])/XYZ.shape[2]
        # XYZ[2,:] = XYZ[2,:]-sum(XYZ[2,:])/XYZ.shape[2]
        # XYZ = squeeze(XYZ).T
        # XYZ = nan_to_num(XYZ)
    
        # canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        # view = canvas.central_widget.add_view()
        # # create scatter object and fill in the data
        # scatter = visuals.Markers()
        # scatter.set_data(XYZ, edge_color=(1,1,1), face_color=(1, 1, 1, .5), size=3)
        # view.add(scatter)
        # axis = visuals.XYZAxis(parent=view.scene)
        # view.camera = 'arcball'  # or try 'arcball'

    def loaddata(self):
        # load calib data
        lightdata = loadmat(self.tagetfolder+'/light.mat')  # load light.mat file
        S = lightdata['S']
        K = lightdata['K']
        Phi = lightdata['Phi']
        Dir = lightdata['Dir']
        mu = lightdata['mu']
        cameradata = loadmat(self.tagetfolder+'/camera.mat')  # load camera.mat file
        K = cameradata['K']
        self.calib = {'S':S, 'Dir':Dir, 'mu':mu, 'Phi':Phi, 'K':K}
        self.nimgs = S.shape[0]
        self.logger.debug("calibration data loading completed")
        # load parameters setting
        self.params = {}
        self.params['z0'] = 700
        self.params['maxit'] = 100
        self.params['estimator'] = 'LS' 
        self.params['indices'] = np.arange(1,self.nimgs-2)
        self.params['ratio'] = 10
        self.params['self_shadows'] = 0
        self.logger.debug("parameters setting completed")
        # load image
        Iamb = Image.open(self.tagetfolder+'/ambient.tif')
        Iamb = np.asarray(Iamb,dtype=np.uint16)
        mask = Image.open(self.tagetfolder+'/mask.tif')
        mask = np.asarray(mask,dtype=np.uint16)
        self.nrows = Iamb.shape[0]
        self.ncols = Iamb.shape[1]

        [xx,yy] = np.meshgrid(np.arange(1,self.ncols+1),np.arange(1,self.nrows+1))
        xx = xx - self.calib['K'][0,2]
        yy = yy - self.calib['K'][1,2]
        ff = (self.calib['K'][0,0]+self.calib['K'][1,1])/2
        cos4a = pow((ff/np.sqrt(xx*xx+yy*yy+ff*ff)),4)
        I = np.zeros((self.nrows,self.ncols,self.nchannels,self.nimgs))
        for i in range(0,self.nimgs):
            Ii = Image.open(self.tagetfolder+'/%04d.tif'%(i+1))
            Ii = np.asarray(Ii,dtype=np.uint16)
            for ch in range(self.nchannels):
                Ii[:,:,ch] = (Ii-Iamb)[:,:,ch]/cos4a
                I[:,:,ch,i] = Ii[:,:,ch]
        
        self.data = {}
        self.data['I'] = I
        self.data['mask'] = mask
        self.data['nrows'] = self.nrows
        self.data['ncols'] = self.ncols
        self.data['nimgs'] = self.nimgs
        self.logger.debug("image load completed")

if __name__ == "__main__":
    if sys.flags.interactive != 1:
        a = Fcn_NPS("./Example_Img/Buddha")
        vispy.app.run()
