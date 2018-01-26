#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
from numpy import *
from scipy.sparse import csr_matrix
import numpy.matlib as matlib
import gc


class Near_Ps(object):
    """docstring for Near_Ps."""

    def __init__(self):
        super(Near_Ps, self).__init__()
        self.mask = ones((800, 800), dtype=bool)
        self.S = array([[-219.439438509374, -57.9176650067909, 517.009286741167],
                        [-206.718243107626, -185.996118240189, 402.103795661335],
                        [-218.391677320354, 46.7271522090837, 397.549748171286],
                        [21.7910301028015, -159.960595417211, 393.472594621813],
                        [2.38654996705536, 110.471407322805, 348.245256050471],
                        [213.068874525323, -181.943871344583, 444.703257457446],
                        [216.880699563198, 11.5862355072107, 452.368512857672],
                        [212.426602757410, -79.2087389806415, 505.618362417849]])
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        self.nchannels = 3
        self.nimgs = 8
        z = 0
        self.K = array([[4092.66394448750, 0, 1244.12180883431], [
            0, 4097.97886076197, 903.583728699309], [0, 0, 1]])
        self.lambda_ = 0
        self.shadows = True
        self.I = random.rand(self.nimgs, self.nchannels,
                             self.nrows, self.ncols)
        self.Phi = array([54229968.5201271, 40943902.7144921, 39176492.0197441, 34748554.9936707,
                          43119457.2127401, 39409192.5974319, 47548583.0849450, 32084397.6480282])
        self.Dir = array([[0.964202201085580, -0.102080295424644, 0.244731952772294],
                          [0.984520205216383, 0.119643201794824,
                           0.128083838890681],
                          [0.829588771768258, -0.515986409987597,
                           0.213402189454898],
                          [0.0829830287569756, 0.689052164903222,
                           0.719945088864770],
                          [0.213209288081382, -0.786688977364903,
                           0.579363661587775],
                          [-0.664171608196248, 0.583892983981335,
                           0.466845861204083],
                          [-0.881106133645938, -0.278186582308633,
                           0.382445037456298],
                          [-0.934949119875948, -0.132485943880536, 0.329116420005557]])
        self.mu = array([1, 1, 1, 1, 1, 1, 1, 1])
        self.fx = 0
        self.fy = 0
        self.x0 = 0
        self.y0 = 0
        self.z = ones((10, 10)) * 700
        self.npix = 0

        self.S = mat(self.S)
        self.K = mat(self.K)
        self.Phi = mat(self.Phi).T
        self.Dir = mat(self.Dir)
        self.mu = mat(self.mu).T

    def main_function(self):
        # Intrinsics
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.x0 = self.K[0, 2]
        self.y0 = self.K[1, 2]

        # Dx, Dy =
        self.make_gradient()
        # Divide images by lighting intensities and normalize to [0,1]
        if(self.nchannels == 1):
            for i in range(self.nimgs):
                self.I[i] = self.I[i] / self.Phi[i, 0]
        else:
            if(self.Phi.shape[1] == 1):
                self.Phi = hstack((self.Phi, self.Phi, self.Phi))
            for i in range(self.nimgs):
                for ch in range(self.nchannels):
                    self.I[i, ch] = self.I[i, ch] / self.Phi[i, ch]

        max_I = amax(self.I)
        self.I = self.I / max_I
        # Scaled pixel units
        uu, vv = meshgrid(range(self.ncols), range(self.nrows))
        u_tilde = (uu - self.x0)
        v_tilde = (vv - self.y0)
        del self.x0, self.y0, uu, vv
        gc.collect()

        # # Use a bounding box
        imask = where(self.mask > 0)
        imin = min(imask[0])
        imax = max(imask[0])
        jmin = min(imask[1])
        jmax = max(imask[1])
        self.z = self.z[imin:imax + 1, jmin:jmax + 1]
        u_tilde = u_tilde[imin:imax + 1, jmin:jmax + 1]
        v_tilde = v_tilde[imin:imax + 1, jmin:jmax + 1]
        self.I = self.I[:, :, imin:imax + 1, jmin:jmax + 1]
        self.mask = self.mask[imin:imax + 1, jmin:jmax + 1]
        imask = where(self.mask > 0)
        self.npix = len(imask[0])
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        del imin, imax, jmin, jmax
        gc.collect()
        #
        # # Some useful variables
        # px_rep = matlib.repmat(u_tilde[imask], self.nimgs, 1).T
        # py_rep = matlib.repmat(v_tilde[imask], self.nimgs, 1).T
        # Dx_rep = matlib.repmat(Dx, self.nimgs, 1)
        # Dy_rep = matlib.repmat(Dy, self.nimgs, 1)
        #
        # # Vectorize data
        # self.I = self.I[:, :, imask[0], imask[1]]
        # self.I = self.I.reshape(
        #     (self.nimgs, self.nchannels, -1))
        # self.I = swapaxes(self.I, 0, 1)
        #
        # # Sort images to remove shadows and highlights
        # W_idx = self.sort_linear_index(1, "descend")

        #########################################
        # Initialize variables
        # self.z[self.mask == 0] = nan
        # z_tilde = (log(self.z[imask]))
        # rho = ones([self.nrows, self.ncols, self.nchannels]) / max_I
        # XYZ = array([[self.z * u_tilde], [self.z * v_tilde], [self.z]])
        # zx = Dx * mat(z_tilde).T
        # zy = Dy * mat(z_tilde).T
        # Nx = zeros([self.nrows, self.ncols])
        # Ny = zeros([self.nrows, self.ncols])
        # Nz = zeros([self.nrows, self.ncols])
        # Nx[imask] = self.fx * zx.T
        # Ny[imask] = self.fy * zy.T
        # Nz[imask] = -u_tilde[imask] * zx - v_tilde[imask] * zy - 1
        # dz = sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
        # N = [Nx / dz, Ny / dz, Nz / dz]
        # if(self.nchannels == 1):
        #     rho_tilde = dz.reshape((self.nrows, self.ncols, self.nchannels))
        # else:
        #     rho_tilde = array([[dz.T], [dz.T], [dz.T]]).reshape(
        #         (self.nrows, self.ncols, self.nchannels))
        # rho_tilde = (rho / rho_tilde).reshape((self.nrows *
        #                                        self.ncols, self.nchannels))
        # rho_tilde = rho_tilde[imask[0] * self.ncols + imask[1], :]
        # tab_nrj = array([])

        ########################################
        # Initial energy
        # energy = 0
        # Tz, grad_Tz = self.t_fcn(
        #     z_tilde, u_tilde[imask] / self.fx, v_tilde[imask] / self.fy)
        # psi = self.shading_fcn(z_tilde, Tz, px_rep, py_rep, Dx_rep, Dy_rep)
        # psi = psi.reshape((self.npix, self.nimgs))
        # for ch in range(self.nchannels):
        #     Ich = self.I[ch]
        #     Wch_idx = W_idx[ch]
        #     energy = energy
        #     self.J_fcn(rho_tilde[:, ch], multiply(Wch_idx,(psi.T)), multiply(Wch_idx,(Ich)))

    def test(self):
        # self.mask[:, 0] = 0
        # self.mask[1, 1] = 0
        self.main_function()
        a = array([1, 0, 0])
        b = random.random((3, 8))

    def make_gradient(self):
        Dyp, Dym, Dxp, Dxm, Omega, index_matrix =self.graddient_operators()
        Dy = Dyp
        no_bottom = where(~Omega[0])
        no_bottom = index_matrix[no_bottom][nonzero(index_matrix[no_bottom])].astype('int32')
        print Dy
        # Dy[no_bottom,:] = Dym[no_bottom, :]
        #
        # Dx = Dxp
        # no_right = where(~Omega[2])
        # no_right = index_matrix[no_right][nonzero(index_matrix[no_right])]
        # Dx[no_right.astype('int32')] = Dxm[no_right.astype('int32'), :]
        #
        # return Dx, Dy

    def graddient_operators(self):
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        Omega_padded = pad(self.mask, 1, 'constant')

        Omega = [0] * 4
        # Pixels who have bottom neighbor in mask
        Omega[0] = self.mask * Omega_padded[2:, 1:-1]
        # Pixels who have top neighbor in mask
        Omega[1] = self.mask * Omega_padded[:-2, 1:-1]
        # Pixels who have right neighbor in mask
        Omega[2] = self.mask * Omega_padded[1:-1, 2:]
        # Pixels who have left neighbor in mask
        Omega[3] = self.mask * Omega_padded[1:-1, :-2]

        imask = where(self.mask.T > 0)
        index_matrix = zeros([self.nrows, self.ncols]).T
        index_matrix[imask] = range(len(imask[0]))
        index_matrix = index_matrix.T

        # Dv matrix
        # When there is a neighbor on the right : forward differences
        idx_c = where(Omega[2] > 0)
        indices_centre = index_matrix[idx_c].astype('int32')
        indices_right = index_matrix[(idx_c[0], idx_c[1] + 1)].astype('int32')
        II = indices_centre
        JJ = indices_right
        KK = ones(len(indices_centre))
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack((KK, -KK))

        Dvp = csr_matrix((KK,(II,JJ)),shape=(len(imask[0]),len(imask[0])))

        # When there is a neighbor on the left : backword differences
        idx_c = where(Omega[3] > 0)
        indices_centre = index_matrix[idx_c].astype('int32')
        indices_right = index_matrix[(idx_c[0], idx_c[1] - 1)].astype('int32')
        II = indices_centre
        JJ = indices_right
        KK = -ones([len(indices_centre)])
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack((KK, -KK))

        Dvm = csr_matrix((KK,(II,JJ)),shape=(len(imask[0]),len(imask[0])))

        # When there is a neighbor on the bottom : forward differences
        idx_c = where(Omega[0] > 0)
        indices_centre = index_matrix[idx_c].astype('int32')
        indices_right = index_matrix[(idx_c[0] + 1, idx_c[1])].astype('int32')
        II = indices_centre
        JJ = indices_right
        KK = ones([len(indices_centre)])
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack([KK, -KK])

        Dup = csr_matrix((KK,(II,JJ)),shape=(len(imask[0]),len(imask[0])))

        # When there is a neighbor on the top : backword differences
        idx_c = where(Omega[1] > 0)
        indices_centre = index_matrix[idx_c].astype('int32')
        indices_right = index_matrix[(idx_c[0] - 1, idx_c[1])].astype('int32')
        II = indices_centre
        JJ = indices_right
        KK = -ones([len(indices_centre)])
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack([KK, -KK])

        Dum = csr_matrix((KK,(II,JJ)),shape=(len(imask[0]),len(imask[0])))

        Omega = array(Omega)
        Omega.astype(bool)

        return Dup, Dum, Dvp, Dvm, Omega, index_matrix

    def sort_linear_index(self, sortDim, sortOrder):
        self.npix = self.I.shape[2]
        W_idx = zeros(self.I.shape)
        indices = array([1, 2, 3, 4, 5])
        W_idx[:, indices, :] = 1

        for ch in range(self.nchannels):
            for i in range(self.npix):
                if (sortOrder == "descend"):
                    W_idx[ch, :, i] = W_idx[ch, :, i][7 -
                                                      argsort(-self.I[ch, :, i])]
                else:
                    W_idx[ch, :, i] = W_idx[ch, :,
                                            i][argsort(self.I[ch, :, i])]

        return W_idx

    def phi_fcn(self, x):
        if(self.estimator == 'LS'):
            return x * x
        elif(self.estimator == 'Cauchy'):
            return self.lambda_ * self.lambda_ * log(1 + (x * x) / (self.lambda_ * self.lambda_))
        elif(self.estimator == 'Lp'):
            return pow(abs(x), self.lambda_)
        elif(self.estimator == 'GM'):
            return x * x / (x * x + self.lambda_ * self.lambda_)
        elif(self.estimator == 'Welsh'):
            return self.lambda_ * self.lambda_ * (1 - exp(-x * x / self.lambda_ * self.lambda_))
        elif(self.estimator == 'Tukey'):
            return (abs(x) <= self.lambda_) * (1 - pow(1 - x * x / (self.lambda_ * self.lambda_), 3)) * self.lambda_ * self.lambda_ + (asb(x) < self.lambda_) * (self.lambda_ * self.lambda_)

    def w_fcn(self, x):
        if(self.estimator == 'LS'):
            return 2 * ones(x.shape)
        elif(self.estimator == 'Cauchy'):
            return 2 * self.lambda_ * self.lambda_ / (x * x + self.lambda_ * self.lambda_)
        elif(self.estimator == 'Lp'):
            return pow(self.lambda_ * (max(5e-3, abs(x))), self.lambda_ - 2)
        elif(self.estimator == 'GM'):
            return 2 * self.lambda_ * self.lambda_ / pow((x * x) + self.lambda_ * self.lambda_, 2)
        elif(self.estimator == 'Welsh'):
            return 2 * exp(-x * x / (self.lambda_ * self.lambda_))
        elif(self.estimator == 'Tukey'):
            return 6 * (abs(x) <= self.lambda_) * pow(1 - x * x / (self.lambda_ * self.lambda_), 2)

    def psi_fcn(self, x):
        if(self.shadows):
            return fmax(x, 0)
        else:
            x

    def chi_fcn(self, x):
        if(self.shadows):
            return (x >= 0) * ones(x.shape)
        else:
            return 1 * ones(x.shape)

    def shading_fcn(self, z, tz, px_rep, py_rep, Dx_rep, Dy_rep):
        resx = mat(zeros((self.npix * self.nimgs, self.npix * self.nimgs)))
        spdiagsx = (self.fx * tz[:, :, 0] - px_rep * tz[:, :, 2]).reshape(-1)
        for i in range(self.ncols * self.nrows):
            resx[i, i] = spdiagsx[i]
        resx = resx * Dx_rep

        resy = mat(zeros((self.npix * self.nimgs, self.npix * self.nimgs)))
        spdiagsy = (self.fy * tz[:, :, 1] - py_rep * tz[:, :, 2]).reshape(-1)
        for i in range(self.ncols * self.nrows):
            resy[i, i] = spdiagsy[i]
        resy = resy * Dy_rep

        resz = tz[:, :, 2].reshape(-1)

        res = (resx + resy) * mat(z).T - mat(resz).T
        res = array(res)
        return res

    def r_fcn(self, rho, shadz, II):
        shadz = shadz.reshape(-1)
        II = II.reshape(-1)
        res = matlib.repmat(rho,1, self.nimgs).reshape(-1)*self.psi_fcn(shadz) -II
        # print shadz.shape
        # print II.shape
        # print res.shape
        # print self.psi_fcn(shadz).shape
        # return res

    def J_fcn(self, rho, shadz, II):
        self.r_fcn(rho, shadz, II)

    def t_fcn(self, z, u_tilde, v_tilde):
        self.npix = len(z)
        # Current mesh
        exp_z = exp(z)
        XYZ = vstack((exp_z * u_tilde, exp_z * v_tilde, exp_z))
        # T_field
        T_field = zeros((3, self.nimgs, self.npix))
        a_field = zeros((self.nimgs, self.npix))
        da_field = zeros((self.nimgs, self.npix))
        for i in range(self.nimgs):
            # unit lighting filed
            T_field[0, i] = self.S[i, 0] - XYZ[0]
            T_field[1, i] = self.S[i, 1] - XYZ[1]
            T_field[2, i] = self.S[i, 2] - XYZ[2]
            normS_i = sqrt(pow(T_field[0][i], 2) +
                           pow(T_field[1][i], 2) +
                           pow(T_field[2][i], 2))
            # attenuation = anisotropy / squared distance
            scal_prod = -T_field[0, i] * self.Dir[i, 0] - \
                T_field[1, i] * self.Dir[i, 0] - T_field[2, i] * self.Dir[i, 2]
            a_field[i] = pow(scal_prod, self.mu[i, 0]) / \
                pow(normS_i, (3 + self.mu[i, 0]))
            da_field[i] = self.mu[i, 0] * pow(scal_prod, self.mu[i, 0] - 1) * (XYZ[0] * self.Dir[i, 0] + XYZ[1] * self.Dir[i, 1] + XYZ[2] * self.Dir[i, 2]) / (pow(normS_i, self.mu[i, 0])) - (
                self.mu[i, 0] + 3) * pow(scal_prod, self.mu[i, 0]) * (-T_field[0, i] * XYZ[0] - T_field[1, i] * XYZ[1] - T_field[2, i] * XYZ[2]) / pow(normS_i, self.mu[i, 0] + 5)
            # Final lighting field
            T_field[0, i] = T_field[0, i] * a_field[i]
            T_field[1, i] = T_field[1, i] * a_field[i]
            T_field[2, i] = T_field[2, i] * a_field[i]

        grad_t = zeros((3, self.nimgs, self.npix))
        grad_t[0] = (-exp_z * u_tilde) * (a_field +
                                          da_field) + (da_field.T * self.S[:, 0]).T
        grad_t[1] = (-exp_z * v_tilde) * (a_field +
                                          da_field) + (da_field.T * self.S[:, 1]).T
        grad_t[0] = (-exp_z) * (a_field + da_field) + \
            (da_field.T * self.S[:, 2]).T

        T_field = swapaxes(T_field, 0, 2)
        grad_t = swapaxes(grad_t, 0, 2)
        return T_field, grad_t


if __name__ == "__main__":
    a = Near_Ps()
    a.test()
