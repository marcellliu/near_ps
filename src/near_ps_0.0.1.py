#!/usr/bin/python
# -*- coding:utf-8 -*-

import cv2
from numpy import *
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import inv
import numpy.matlib as matlib
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky
from sksparse.cholmod import cholesky_AAt
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import lobpcg
import gc

class Near_Ps(object):
    """docstring for Near_Ps."""

    def __init__(self):
        super(Near_Ps, self).__init__()
        # self.mask = ones((200, 200), dtype=bool)
        self.mask = matrix([[1, 0, 0],
                            [1, 1, 1],
                            [0, 0, 1]])
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
        self.nchannels = 1
        self.nimgs = 8
        z = 0
        self.K = array([[4092.66394448750, 0, 1244.12180883431], [
            0, 4097.97886076197, 903.583728699309], [0, 0, 1]])
        self.lambda_ = 0
        self.shadows = True
        # self.I = random.rand(self.nimgs, self.nchannels,
        # self.nrows, self.ncols)
        self.I = ones((self.nimgs, self.nchannels,
                       self.nrows, self.ncols))
        self.Phi = matrix([54229968.5201271, 40943902.7144921, 39176492.0197441, 34748554.9936707,
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
        self.z = ones((self.nrows, self.ncols)) * 700.0
        self.x = 0
        self.estimator = 'LS'
        self.precond = 'cmg'
        self.semi_calibrated = 0
        self.maxit = 100
        self.tol = 1e-3

        self.Phi = self.Phi.T
        self.mu = self.mu.T

    def main_function(self):
        # Intrinsics
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.x0 = self.K[0, 2]
        self.y0 = self.K[1, 2]

        Dx, Dy = self.make_gradient()

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
        uu, vv = meshgrid(range(1, self.ncols + 1), range(1, self.nrows + 1))
        u_tilde = (uu - self.x0)
        v_tilde = (vv - self.y0)

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

        # Some useful variables
        px_rep = matlib.repmat(u_tilde[imask], self.nimgs, 1).T
        py_rep = matlib.repmat(v_tilde[imask], self.nimgs, 1).T

        Dx = Dx.tocoo()
        Dy = Dy.tocoo()

        data = tile(Dx.data, self.nimgs)
        add_row = arange(self.nimgs).repeat(len(Dx.row)) * Dx.shape[0]
        row = tile(Dx.row, self.nimgs) + add_row
        col = tile(Dx.col, self.nimgs)
        Dx_rep = coo_matrix((data, (row, col)), shape=(
            Dx.shape[0] * self.nimgs, Dx.shape[1]))
        data = tile(Dy.data, self.nimgs)
        add_row = arange(self.nimgs).repeat(len(Dy.row)) * Dy.shape[0]
        row = tile(Dy.row, self.nimgs) + add_row
        col = tile(Dy.col, self.nimgs)
        Dy_rep = coo_matrix((data, (row, col)), shape=(
            Dy.shape[0] * self.nimgs, Dy.shape[1]))

        # Vectorize data
        self.I = self.I[:, :, imask[0], imask[1]]
        self.I = self.I.reshape(
            (self.nimgs, self.nchannels, -1))
        self.I = swapaxes(self.I, 1, 0)
        self.I = swapaxes(self.I, 2, 1)

        # Sort images to remove shadows and highlights
        W_idx = self.sort_linear_index(1, "descend")

        #########################################
        # Initialize variables
        self.z[self.mask == 0] = nan
        z_tilde = (log(self.z[imask]))
        rho = ones([self.nchannels, self.nrows, self.ncols]) / max_I
        XYZ = array([[self.z * u_tilde], [self.z * v_tilde], [self.z]])
        zx = Dx * (z_tilde.T)
        zy = Dy * mat(z_tilde).T
        Nx = zeros([self.nrows, self.ncols])
        Ny = zeros([self.nrows, self.ncols])
        Nz = zeros([self.nrows, self.ncols])
        Nx[imask] = self.fx * zx.T
        Ny[imask] = self.fy * zy.T
        Nz[imask] = -u_tilde[imask] * zx - v_tilde[imask] * zy - 1
        dz = sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
        N = [Nx / dz, Ny / dz, Nz / dz]

        rho_tilde = (rho / dz).T.reshape((self.nrows *
                                          self.ncols, self.nchannels))
        rho = tile(rho_tilde, self.nchannels)
        rho_tilde = rho_tilde[imask[0] + imask[1] * self.ncols, :]

        #####################################
        # Initial energy
        energy = 0
        Tz, grad_Tz = self.t_fcn(
            z_tilde, u_tilde[imask] / self.fx, v_tilde[imask] / self.fy)
        psi = self.shading_fcn(z_tilde, Tz, px_rep, py_rep, Dx_rep, Dy_rep)
        psi = psi.reshape((self.nimgs, self.npix)).T
        for ch in range(self.nchannels):
            Ich = self.I[ch]
            Wch_idx = W_idx[ch]
            energy += self.J_fcn(rho_tilde[:, ch],
                                 multiply(Wch_idx, (psi)), multiply(Wch_idx, (Ich)))
        #####################################
        # Start iteration
        for it in range(1,self.maxit):
            r = zeros((self.nchannels, self.npix, self.nimgs))
            w = zeros((self.nchannels, self.npix, self.nimgs))
            chi = self.chi_fcn(psi)
            phi_chi = psi * chi

            # Pseudo-albedo update
            for ch in range(self.nchannels):
                Ich = self.I[ch]
                r[ch] = self.r_fcn(rho_tilde[:, ch], psi, Ich).reshape(
                    (self.npix, self.nimgs))
                w[ch] = self.w_fcn(r[ch]) * W_idx[ch]
                denom = (w[ch] * pow(phi_chi, 2)).sum(axis=1)
                idx_ok = where(denom > 0)
                rho_tilde[idx_ok, ch] = (
                    w[ch, idx_ok, ] * self.I[ch, idx_ok] * phi_chi[idx_ok]).sum(axis=2) / denom[idx_ok]

            # log-depth update
            rho_rep = zeros((self.npix, self.nimgs * self.nchannels))
            for ch in range(self.nchannels):
                Ich = self.I[ch]
                r[ch] = self.r_fcn(rho_tilde[:, ch], psi, Ich).reshape(
                    (self.npix, self.nimgs))
                rho_rep[:, arange(
                    self.nimgs) + ch * self.nimgs] = matlib.repmat(rho_tilde, 1, self.nimgs)

            r = r.reshape(self.npix, self.nimgs * self.nchannels)
            w = self.w_fcn(r) * W_idx.reshape(self.npix,self.nimgs * self.nchannels)
            D = matlib.repmat(chi, 1, self.nchannels) * pow(rho_rep, 2) * w
            A = spdiags((self.fx * Tz[0] - px_rep * Tz[2]).T.reshape(-1), 0, self.npix * self.nimgs, self.npix * self.nimgs) * Dx_rep + spdiags(
                (self.fy * Tz[1] - py_rep * Tz[2]).T.reshape(-1), 0, self.npix * self.nimgs, self.npix * self.nimgs) * Dy_rep
            data = (spdiags((self.fx * grad_Tz[0] - px_rep * grad_Tz[2]).T.reshape(-1), 0, self.nimgs * self.npix, self.npix * self.nimgs) * Dx_rep + spdiags(
                (self.fy * grad_Tz[1] - py_rep * grad_Tz[2]).T.reshape(-1), 0, self.nimgs * self.npix, self.npix * self.nimgs) * Dy_rep) *z_tilde- grad_Tz[2].T.reshape(-1)
            data = data.T
            II = arange(self.npix*self.nimgs)
            JJ = tile(arange(self.npix),self.nimgs)
            A += coo_matrix((data, (II, JJ)))
            A = A.tocoo()
            data = tile(A.data, self.nchannels)
            add_row = arange(self.nchannels).repeat(len(A.row)) * A.shape[0]
            row = tile(A.row, self.nchannels) + add_row
            col = tile(A.col, self.nchannels)
            A = coo_matrix((data, (row, col)), shape=(
                Dy.shape[0] * self.nimgs, Dy.shape[1]))
            M = (A.T)*spdiags(D.T.reshape(-1),0,self.nimgs*self.npix*self.nchannels,self.nimgs*self.npix*self.nchannels)*A
            rhs = matlib.repmat(chi,1,self.nchannels)*rho_rep*(rho_rep*(-matlib.repmat(psi,1,self.nchannels))+self.I.reshape(self.npix,self.nimgs*self.nchannels))*w
            rhs = mat((A.T)*rhs.T.reshape(-1)).T

            if(it%5 == 1):
                precond = cholesky(M.tocsc())
            precond.cholesky_inplace(M)
            z_tilde = array(z_tilde+precond.solve_A(rhs).T).reshape(-1)

            # Auxiliary variables
            self.z[imask] = exp(z_tilde)
            zx = (array(z_tilde)*(Dx.T)).reshape(-1)
            zy = (array(z_tilde)*(Dy.T)).reshape(-1)
            Nx = zeros((self.nrows,self.ncols))
            Ny = zeros((self.nrows,self.ncols))
            Nz = zeros((self.nrows,self.ncols))
            Nx[imask] = self.fx*zx
            Ny[imask] = self.fy*zy
            Nz[imask] = -u_tilde[imask]*zx-v_tilde[imask]*zy-1
            dz = sqrt(Nx*Nx+Ny*Ny+Nz*Nz)
            XYZ = array([[self.z * u_tilde/self.fx], [self.z * v_tilde/self.fy], [self.z]])

            # Convergence test
            energy_new = 0
            [Tz,grad_Tz] = self.t_fcn(z_tilde,u_tilde[imask]/self.fx,v_tilde[imask]/self.fy)
            psi = self.shading_fcn(z_tilde,Tz, px_rep, py_rep, Dx_rep, Dy_rep)
            psi = psi.reshape((self.nimgs, self.npix)).T
            for ch in range(self.nchannels):
                Ich = self.I[ch]
                Wch_idx = W_idx[ch]
                energy_new += self.J_fcn(rho_tilde[:, ch], multiply(Wch_idx, (psi)), multiply(Wch_idx, (Ich)))
            relative_diff = abs(energy_new-energy)/energy_new
            energy = energy_new


            if(relative_diff<self.tol):
                break

        return XYZ

    def test(self):
        # self.mask[:, 0] = 0
        # self.mask[1, 1] = 0
        XYZ = self.main_function()
        print XYZ
    # c

    def make_gradient(self):
        Dyp, Dym, Dxp, Dxm, Omega, index_matrix = self.graddient_operators()

        Dy = Dyp
        no_bottom = where(~Omega[0])
        no_bottom = index_matrix[no_bottom][nonzero(
            index_matrix[no_bottom])].astype('int32') - 1
        Dy[no_bottom, :] = Dym[no_bottom, :]

        Dx = Dxp
        no_right = where(~Omega[2])
        no_right = index_matrix[no_right][nonzero(
            index_matrix[no_right])].astype('int32') - 1
        Dx[no_right, :] = Dxm[no_right, :]
        # print Dx.todense()

        return Dx, Dy
    # c

    def graddient_operators(self):
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        Omega_padded = pad(self.mask, 1, 'constant')

        Omega = [0] * 4
        # Pixels who have bottom neighbor in mask
        Omega[0] = multiply(self.mask, Omega_padded[2:, 1:-1]).astype(bool)
        # Pixels who have top neighbor in mask
        Omega[1] = multiply(self.mask, Omega_padded[:-2, 1:-1]).astype(bool)
        # Pixels who have right neighbor in mask
        Omega[2] = multiply(self.mask, Omega_padded[1:-1, 2:]).astype(bool)
        # Pixels who have left neighbor in mask
        Omega[3] = multiply(self.mask, Omega_padded[1:-1, :-2]).astype(bool)

        imask = where(self.mask.T > 0)
        index_matrix = zeros([self.nrows, self.ncols]).T
        index_matrix[imask] = range(1, 1 + len(imask[0]))
        index_matrix = index_matrix.T

        # Dv matrix
        # When there is a neighbor on the right : forward differences
        idx_c = where(Omega[2] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0], idx_c[1] + 1)].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = ones(len(indices_centre))
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack((KK, -KK))

        # Dvp = csr_matrix((KK,(II,JJ)),shape=(len(imask[0]),len(imask[0])))
        Dvp = lil_matrix((len(imask[0]), len(imask[0])))
        Dvp[II, JJ] = KK

        # When there is a neighbor on the left : backword differences
        idx_c = where(Omega[3] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0], idx_c[1] - 1)].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = -ones([len(indices_centre)])
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack((KK, -KK))

        Dvm = lil_matrix((len(imask[0]), len(imask[0])))
        Dvm[II, JJ] = KK

        # When there is a neighbor on the bottom : forward differences
        idx_c = where(Omega[0] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0] + 1, idx_c[1])].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = ones([len(indices_centre)])
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack([KK, -KK])

        Dup = lil_matrix((len(imask[0]), len(imask[0])))
        Dup[II, JJ] = KK

        # When there is a neighbor on the top : backword differences
        idx_c = where(Omega[1] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0] - 1, idx_c[1])].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = -ones([len(indices_centre)])
        II = hstack((II, indices_centre))
        JJ = hstack((JJ, indices_centre))
        KK = hstack([KK, -KK])

        Dum = lil_matrix((len(imask[0]), len(imask[0])))
        Dum[II, JJ] = KK

        return Dup, Dum, Dvp, Dvm, Omega, index_matrix
    # c

    def sort_linear_index(self, sortDim, sortOrder):
        W_idx = zeros(self.I.shape)
        indices = array([1, 2, 3, 4, 5])
        W_idx[:, :, indices] = 1
        sortIndex = argsort(-self.I)
        sortindexq = unravel_index(sortIndex, self.I.shape)
        return W_idx
    # c

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
    # c

    def psi_fcn(self, x):
        if(self.shadows):
            return fmax(x, 0)
        else:
            x
    # c

    def chi_fcn(self, x):
        if(self.shadows):
            return (x >= 0) * ones(x.shape)
        else:
            return 1 * ones(x.shape)
    # c

    def shading_fcn(self, z, tz, px_rep, py_rep, Dx_rep, Dy_rep):
        resx = mat(zeros((self.npix * self.nimgs, self.npix * self.nimgs)))
        spdiagsx = (self.fx * tz[0, :, :] - px_rep * tz[2, :, :]).T.reshape(-1)
        spdiagsx = spdiags(spdiagsx, 0, self.npix *
                           self.nimgs, self.npix * self.nimgs)
        resx = spdiagsx * Dx_rep

        resy = mat(zeros((self.npix * self.nimgs, self.npix * self.nimgs)))
        spdiagsy = (self.fy * tz[1, :, :] - py_rep * tz[2, :, :]).T.reshape(-1)
        spdiagsy = spdiags(spdiagsy, 0, self.npix *
                           self.nimgs, self.npix * self.nimgs)
        resy = spdiagsy * Dy_rep

        resz = tz[2, :, :].T.reshape(-1)

        res = (resx + resy) * z - resz
        return res
    # c

    def r_fcn(self, rho, shadz, II):
        II = II.T.reshape(-1)
        shadz = shadz.T.reshape(-1)
        res = matlib.repmat(rho, 1, self.nimgs) * \
            self.psi_fcn(shadz) - II
        return res
    # c

    def J_fcn(self, rho, shadz, II):
        res = self.r_fcn(rho, shadz, II)
        res = self.phi_fcn(res)
        res = sum(res)
        return res
    # c

    def t_fcn(self, z, u_tilde, v_tilde):
        self.npix = len(z)
        # Current mesh
        exp_z = exp(z)
        XYZ = vstack((exp_z * u_tilde, exp_z * v_tilde, exp_z)).T
        # T_field
        T_field = zeros((3, self.npix, self.nimgs))
        a_field = zeros((self.npix, self.nimgs))
        da_field = zeros((self.npix, self.nimgs))
        for i in range(self.nimgs):
            # unit lighting filed
            T_field[0, :, i] = self.S[i, 0] - XYZ[:, 0]
            T_field[1, :, i] = self.S[i, 1] - XYZ[:, 1]
            T_field[2, :, i] = self.S[i, 2] - XYZ[:, 2]
            normS_i = sqrt(pow(T_field[0, :, i], 2) +
                           pow(T_field[1, :, i], 2) +
                           pow(T_field[2, :, i], 2))
            # attenuation = anisotropy / squared distance
            scal_prod = -T_field[0, :, i] * self.Dir[i, 0] - \
                T_field[1, :, i] * self.Dir[i, 1] - \
                T_field[2, :, i] * self.Dir[i, 2]
            a_field[:, i] = pow(scal_prod, self.mu[i]) / \
                pow(normS_i, (3 + self.mu[i]))
            da_field[:, i] = self.mu[i] * pow(scal_prod, self.mu[i] - 1) * (XYZ[:, 0] * self.Dir[i, 0] + XYZ[:, 1] * self.Dir[i, 1] + XYZ[:, 2] * self.Dir[i, 2]) / (pow(normS_i, self.mu[i] + 3)) - (
                self.mu[i] + 3) * pow(scal_prod, self.mu[i]) * (-T_field[0, :, i] * XYZ[:, 0] - T_field[1, :, i] * XYZ[:, 1] - T_field[2, :, i] * XYZ[:, 2]) / pow(normS_i, self.mu[i] + 5)
            # Final lighting field
            T_field[0, :, i] = T_field[0, :, i] * a_field[:, i]
            T_field[1, :, i] = T_field[1, :, i] * a_field[:, i]
            T_field[2, :, i] = T_field[2, :, i] * a_field[:, i]

        grad_t = zeros((3, self.npix, self.nimgs))
        grad_t[0] = ((-exp_z * u_tilde).T * (a_field +
                                             da_field).T).T + (self.S[:, 0] * da_field)
        grad_t[1] = ((-exp_z * v_tilde).T * (a_field +
                                             da_field).T).T + (self.S[:, 1] * da_field)
        grad_t[2] = (-exp_z.T * (a_field + da_field).T).T + \
            (self.S[:, 2] * da_field)

        return T_field, grad_t

if __name__ == "__main__":
    a = Near_Ps()
    a.test()
