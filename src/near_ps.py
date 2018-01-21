#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
import numpy.matlib
import gc

class Near_Ps(object):
    """docstring for Near_Ps."""

    def __init__(self):
        super(Near_Ps, self).__init__()
        self.mask = np.ones((3, 3))
        self.S = np.zeros((8, 3))
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        self.nchannels = 3
        self.nimgs = 8
        self.z = 0
        self.K = np.array([[4092.66394448750, 0, 1244.12180883431], [
                          0, 4097.97886076197, 903.583728699309], [0, 0, 1]])
        self.lambda_ = 0
        self.shadows = True
        self.I = np.ones((8, 3, 3, 3))
        self.Phi = np.array([[54229968.5201271],[40943902.7144921],[39176492.0197441],[34748554.9936707],[43119457.2127401],[39409192.5974319],[47548583.0849450],[32084397.6480282]])

    def main_function(self):
        fx = self.K[0][0]
        fy = self.K[1][1]
        x0 = self.K[0][2]
        y0 = self.K[1][2]

        if(self.nchannels == 1):
            for i in range(self.nimgs):
                self.I[i] = self.I[i]/self.Phi[i]
        else:
            if(self.Phi.shape[1] == 1):
                self.Phi = np.hstack((self.Phi,self.Phi,self.Phi))
            for i in range(self.nimgs):
                for ch in range(self.nchannels):
                    self.I[i,ch] = self.I[i,ch]/self.Phi[i,ch]

        max_I = np.amax(self.I)
        self.I = self.I/max_I

        # Scaled pixel units
        uu, vv = np.meshgrid(range(self.ncols), range(self.nrows))
        u_tilde = (uu-x0)
        v_tilde = (vv-y0)
        del x0,y0,uu,vv
        gc.collect()

        # Use a bounding box
        imask = np.where(self.mask > 0)
        imin = min(imask[0])
        imax = max(imask[0])
        jmin = min(imask[1])
        jmax = max(imask[1])
        self.z = self.z[imin:imax+1,jmin:jmax+1]
        u_tilde = u_tilde[imin:imax+1,jmin:jmax+1]
        v_tilde = v_tilde[imin:imax+1,jmin:jmax+1]
        self.I= self.I[:,:,imin:imax+1,jmin:jmax+1]
        self.mask = self.mask[imin:imax+1,jmin:jmax+1]
        imask = np.where(self.mask > 0)
        npix = len(imask[0])
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        del imin,imax,jmin,jmax
        gc.collect()

        # Some useful variables
        px_rep = np.matlib.repmat(u_tilde[imask],1,self.nimgs)
        py_rep = np.matlib.repmat(v_tilde[imask],1,self.nimgs)
        # Dx_rep = np.matlib.repmat(u_tilde[imask],self.nimgs,1)
        # Dy_rep = np.matlib.repmat(u_tilde[imask],self.nimgs,1)

        # Vectorize data
        self.I = self.I.reshape((self.nimgs,self.nchannels,self.nrows*self.ncols))
        self.I = np.swapaxes(self.I,0,1)
        self.I = self.I[:,:,[imask[0]*self.ncols+imask[1]]]
        # Sort images to remove shadows and highlights

    def test(self):
        # imask = np.where(self.mask > 0)
        # index_matrix = np.zeros([3, 3])
        # index_matrix[imask] = range(1, len(imask[0]) + 1)
        # # print index_matrix
        a = np.array([[1, 2],[ 3, 4]])
        self.mask[:,0]=0
        self.mask[1,1] =0
        self.z = np.ones((10, 10))*70
        self.main_function()

    def make_gradient(self):
        Dyp, Dym, Dxp, Dxm = self.graddient_operators()
        Dy = Dyp
        # no_bottom =

    def graddient_operators(self):
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        Omega_padded = np.lib.pad(self.mask, 1, 'constant')

        Omega = [0] * 4
        # Pixels who have bottom neighbor in mask
        Omega[0] = self.mask * Omega_padded[2:, 1:-1]
        # Pixels who have top neighbor in mask
        Omega[1] = self.mask * Omega_padded[:-2, 1:-1]
        # Pixels who have right neighbor in mask
        Omega[2] = self.mask * Omega_padded[1:-1, 2:]
        # Pixels who have left neighbor in mask
        Omega[3] = self.mask * Omega_padded[1:-1, :-2]

        imask = np.where(self.mask > 0)
        index_matrix = np.zeros([self.nrows, self.ncols])
        index_matrix[imask] = range(1, len(imask[0]) + 1)

        # Dv matrix
        # When there is a neighbor on the right : forward differences
        idx_c = np.where(Omega[2] > 0)
        indices_centre = index_matrix[idx_c]
        indices_right = index_matrix[(idx_c[0], idx_c[1] + 1)]
        II = indices_centre
        JJ = indices_right
        KK = np.ones([1, len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack((KK, -KK))

        Dvp = [[II, JJ], KK]
        # When there is a neighbor on the left : backword differences
        idx_c = np.where(Omega[3] > 0)
        indices_centre = index_matrix[idx_c]
        indices_right = index_matrix[(idx_c[0], idx_c[1] - 1)]
        II = indices_centre
        JJ = indices_right
        KK = np.ones([1, len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack((KK, -KK))

        Dvm = [[II, JJ], KK]
        # When there is a neighbor on the bottom : forward differences
        idx_c = np.where(Omega[0] > 0)
        indices_centre = index_matrix[idx_c]
        indices_right = index_matrix[(idx_c[0] + 1, idx_c[1])]
        II = indices_centre
        JJ = indices_right
        KK = np.ones([1, len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack((KK, -KK))

        Dup = [[II, JJ], KK]
        # When there is a neighbor on the top : backword differences
        idx_c = np.where(Omega[0] > 0)
        indices_centre = index_matrix[idx_c]
        indices_right = index_matrix[(idx_c[0] + 1, idx_c[1])]
        II = indices_centre
        JJ = indices_right
        KK = np.ones([1, len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack((KK, -KK))

        Dum = [[II, JJ], KK]
        return Dup, Dum, Dvp, Dvm

    def phi_fcn(self, x):
        if(self.estimator == 'LS'):
            return x * x
        elif(self.estimator == 'Cauchy'):
            return self.lambda_ * self.lambda_ * np.log(1 + (x * x) / (self.lambda_ * self.lambda_))
        elif(self.estimator == 'Lp'):
            return pow(abs(x), self.lambda_)
        elif(self.estimator == 'GM'):
            return x * x / (x * x + self.lambda_ * self.lambda_)
        elif(self.estimator == 'Welsh'):
            return self.lambda_ * self.lambda_ * (1 - np.exp(-x * x / self.lambda_ * self.lambda_))
        elif(self.estimator == 'Tukey'):
            return (abs(x) <= self.lambda_) * (1 - pow(1 - x * x / (self.lambda_ * self.lambda_), 3)) * self.lambda_ * self.lambda_ + (asb(x) < self.lambda_) * (self.lambda_ * self.lambda_)

    def w_fcn(self, x):
        if(self.estimator == 'LS'):
            return 2 * np.ones(x.shape)
        elif(self.estimator == 'Cauchy'):
            return 2 * self.lambda_ * self.lambda_ / (x * x + self.lambda_ * self.lambda_)
        elif(self.estimator == 'Lp'):
            return pow(self.lambda_ * (max(5e-3, abs(x))), self.lambda_ - 2)
        elif(self.estimator == 'GM'):
            return 2 * self.lambda_ * self.lambda_ / pow((x * x) + self.lambda_ * self.lambda_, 2)
        elif(self.estimator == 'Welsh'):
            return 2 * np.exp(-x * x / (self.lambda_ * self.lambda_))
        elif(self.estimator == 'Tukey'):
            return 6 * (abs(x) <= self.lambda_) * pow(1 - x * x / (self.lambda_ * self.lambda_), 2)

    def psi_fcn(self, x):
        if(self.shadows):
            return max(x, 0)
        else:
            x

    def chi_fcn(self, x):
        if(self.shadows):
            return (x >= 0) * np.ones(x.shape)
        else:
            return 1 * np.ones(x.shape)

    def t_fcn(self, Dir, mu, u_tilde, v_tilde):
        npix = len(self.z)
        # Current mesh
        exp_z = np.exp(z)
        XYZ = np.vstack((exp_z * seu_tilde, exp_z * v_tilde, exp_z))
        # T_field
        T_field = np.zeros(size=(3, self.nimgs, npix))
        a_field = np.zeros(size=(self.nimgs, npix))
        da_field = np.zeros(size=(self.nimgs, npix))
        for i in range(self.nimgs):
            # unit lighting filed
            T_field[0][i] = self.S[0][i] - XYZ[0]
            T_field[1][i] = self.S[1][i] - XYZ[1]
            T_field[2][i] = self.S[2][i] - XYZ[2]
            normS_i = np.sqrt(pow(T_field[0][i], 2) +
                              pow(T_field[1][i], 2) +
                              pow(T_field[2][i], 2))
            # attenuation = anisotropy / squared distance
            scal_prod = -T_field[0][i] * Dir[0][i] - \
                T_field[1][i] * Dir[1][i] - T_field[2][i] * Dir[2][i]
            a_field[i] = pow(scal_prod, mu[i]) / pow(normS_i, (3 + mu[i]))
            da_field[i] = mu[i] * (scal_prod, mu[i] - 1) * (XYZ[0] * Dir[0][i] + XYZ[1] * Dir[1][i] + XYZ[2] * Dir[2][i]) / (pow(normS_i, mu[i])) - (
                mu[i] + 3) * pow(scal_prod, mu[i]) * (-T_field[0][i] * XYZ[0] - T_field[1][i] * XYZ[1] - T_field[2][i] * XYZ[2]) / pow(normS_i, mu[i] + 5)
            # Final lighting field
            T_field[0][i] = T_field[0][i] * a_field[i]
            T_field[1][i] = T_field[1][i] * a_field[i]
            T_field[2][i] = T_field[2][i] * a_field[i]

        grad_t = np.zeros(size=(3, self.nimgs, npix))
        grad_t[0] = (-exp_z * u_tilde) * (a_field +
                                          da_field) + (da_field) * (self.S[0])
        grad_t[1] = (-exp_z * v_tilde) * (a_field +
                                          da_field) + (da_field) * (self.S[1])
        grad_t[0] = (-exp_z) * (a_field + da_field) + (da_field) * (self.S[2])
        return T_field, grad_t

    def sort_linear_index(sef, A, sortDim, sortOrder):
        sizeA = A.shape
        # np.Sort


if __name__ == "__main__":
    a = Near_Ps()
    a.test()
