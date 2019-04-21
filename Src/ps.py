# -*- coding:utf-8 -*
import numpy as np
import numpy.matlib as matlib
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu


class ps(object):
    '''
    in this version we only take care about the gray-scale image
    therefore the nchannel is set to 1

    parameter:
    mask: nrows*ncols
    I: nrows*ncols*nimgs
    nimgs: number of images
    nrows: number of rows
    ncols: number of colums
    Dir: direction of light sources nimgs*3
    mu: light source factory nimgs*1
    Phi: light source factory nimgs*1
    S: position of light sources nimgs*3
    K: camera internal parameters matrix 3*3
    z0: initial deepth
    ratio: resolution scaling ratio
    maxit: max iteration time
    '''

    def __init__(self, data, calib, params):
        super(ps, self).__init__()
        self.mask = data['mask']
        self.I = data['I']
        self.nimgs = data['nimgs']
        self.nrows = data['nrows']
        self.ncols = data['ncols']
        self.Dir = calib['Dir']
        self.mu = calib['mu']
        self.Phi = calib['Phi']
        self.S = calib['S']
        self.K = calib['K']
        self.z0 = params['z0']
        self.ratio = params['ratio']
        self.maxit = params['maxit']
        self.estimator = params['estimator']
        self.lambda_ = 0
        self.shadows = True

        self.z = np.ones((self.nrows, self.ncols)) * self.z0
        self.semi_calibrated = 0
        self.tol = 1e-3

        if (self.ratio != 0):
            self.I = self.I[0:self.nrows:self.ratio,0:self.ncols:self.ratio, :]
            self.mask = self.mask[0:self.nrows:self.ratio,
                                  0:self.ncols:self.ratio]/255
            self.z = self.z[0:self.nrows:self.ratio, 0:self.ncols:self.ratio]
            self.K[0:2, :] /= self.ratio
            self.nrows = self.I.shape[0]
            self.ncols = self.I.shape[1]

    def main_fcn(self):
        # Intrinsics
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.x0 = self.K[0, 2]
        self.y0 = self.K[1, 2]

        Dx, Dy = self.make_gradient()

        # Divide images by lighting intensities and normalize to [0,1]
        for i in range(self.nimgs):
                self.I[:, :, i] = self.I[:, :, i] / self.Phi[i]

        max_I = np.amax(self.I)+1
        self.I = self.I / max_I

        # Scaled pixel units
        uu, vv = np.meshgrid(range(1, self.ncols + 1),
                             range(1, self.nrows + 1))
        u_tilde = (uu - self.x0)
        v_tilde = (vv - self.y0)
        # Use a bounding box
        imask = np.where(self.mask > 0)
        imin = min(imask[0])
        imax = max(imask[0])
        jmin = min(imask[1])
        jmax = max(imask[1])
        self.z = self.z[imin:imax + 1, jmin:jmax + 1]
        u_tilde = u_tilde[imin:imax + 1, jmin:jmax + 1]
        v_tilde = v_tilde[imin:imax + 1, jmin:jmax + 1]
        self.I = self.I[imin:imax + 1, jmin:jmax + 1, :]
        self.mask = self.mask[imin:imax + 1, jmin:jmax + 1]
        imask = np.array(np.where(self.mask.T > 0))
        imask[[0, 1]] = imask[[1, 0]]
        imask = tuple(imask)
        self.npix = len(imask[0])
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        # Some useful variables
        px_rep = matlib.repmat(u_tilde[imask], self.nimgs, 1).T
        py_rep = matlib.repmat(v_tilde[imask], self.nimgs, 1).T
        Dx = Dx.tocoo()
        Dy = Dy.tocoo()
        data = np.tile(Dx.data, self.nimgs)
        add_row = np.arange(self.nimgs).repeat(len(Dx.row)) * Dx.shape[0]
        row = np.tile(Dx.row, self.nimgs) + add_row
        col = np.tile(Dx.col, self.nimgs)
        Dx_rep = coo_matrix((data, (row, col)), shape=(
            Dx.shape[0] * self.nimgs, Dx.shape[1]))
        data = np.tile(Dy.data, self.nimgs)
        add_row = np.arange(self.nimgs).repeat(len(Dy.row)) * Dy.shape[0]
        row = np.tile(Dy.row, self.nimgs) + add_row
        col = np.tile(Dy.col, self.nimgs)
        Dy_rep = coo_matrix((data, (row, col)), shape=(
            Dy.shape[0] * self.nimgs, Dy.shape[1]))
        # Vectorize data
        self.I = self.I[imask[0], imask[1], :]
        # Sort images to remove shadows and highlights
        W_idx = self.sort_linear_index(1, "descend")

        #########################################
        # Initialize variables
        self.z[self.mask == 0] = np.nan
        z_tilde = (np.log(self.z[imask]))
        rho = np.ones([self.nrows, self.ncols]) / max_I
        X = np.array(self.z * u_tilde)
        Y = np.array(self.z * v_tilde)
        Z = np.array(self.z)
        zx = Dx * (z_tilde.T)
        zy = Dy * np.mat(z_tilde).T
        Nx = np.zeros([self.nrows, self.ncols])
        Ny = np.zeros([self.nrows, self.ncols])
        Nz = np.zeros([self.nrows, self.ncols])
        Nx[imask] = self.fx * zx.T
        Ny[imask] = self.fy * zy.T
        Nz[imask] = -u_tilde[imask] * zx - v_tilde[imask] * zy - 1
        dz = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)+1e-4
        N = [Nx / dz, Ny / dz, Nz / dz]
        rho_tilde = (rho / dz).T.reshape((self.nrows * self.ncols))
        rho_tilde = rho_tilde[imask[0] + imask[1] * self.ncols]
        #####################################
        # Initial energy
        energy = 0
        Tz, grad_Tz = self.t_fcn(
            z_tilde, u_tilde[imask] / self.fx, v_tilde[imask] / self.fy)
        psi = self.shading_fcn(z_tilde, Tz, px_rep, py_rep, Dx_rep, Dy_rep)
        psi = psi.reshape(self.nimgs, self.npix).T
        energy += self.J_fcn(rho_tilde, np.multiply(W_idx,(psi)), np.multiply(W_idx, self.I))

        #####################################
        # # Start iteration
        for it in range(1, self.maxit):
            r = np.zeros((self.npix, self.nimgs))
            w = np.zeros((self.npix, self.nimgs))
            chi = self.chi_fcn(psi)
            phi_chi = psi * chi

            # Pseudo-albedo update
            r = self.r_fcn(rho_tilde, psi, self.I).reshape(self.nimgs, self.npix).T
            w = self.w_fcn(r) * W_idx
            denom = (w[:, :] * pow(phi_chi, 2)).sum(axis=1)
            idx_ok = np.where(denom > 0)
            rho_tilde[idx_ok] = (w[idx_ok, :] * self.I[idx_ok, :] * phi_chi[idx_ok, :]).sum(axis=2) / denom[idx_ok]

            # log-depth update
            rho_rep = np.zeros((self.npix, self.nimgs))
            r = self.r_fcn(rho_tilde, psi, self.I).reshape(self.nimgs, self.npix).T
            rho_rep = matlib.repmat(rho_tilde, self.nimgs, 1).T
            r = r.reshape(self.npix, self.nimgs)
            w = self.w_fcn(r) * W_idx.reshape(self.npix,self.nimgs)
            D = chi * pow(rho_rep, 2) * w
            A = spdiags((self.fx * Tz[0] - px_rep * Tz[2]).T.reshape(-1), 0, self.npix * self.nimgs, self.npix * self.nimgs) * Dx_rep + spdiags(
                (self.fy * Tz[1] - py_rep * Tz[2]).T.reshape(-1), 0, self.npix * self.nimgs, self.npix * self.nimgs) * Dy_rep
            data = (spdiags((self.fx * grad_Tz[0] - px_rep * grad_Tz[2]).T.reshape(-1), 0, self.nimgs * self.npix, self.npix * self.nimgs) * Dx_rep + spdiags(
                (self.fy * grad_Tz[1] - py_rep * grad_Tz[2]).T.reshape(-1), 0, self.nimgs * self.npix, self.npix * self.nimgs) * Dy_rep) * z_tilde - grad_Tz[2].T.reshape(-1)
            data = data.T
            II = np.arange(self.npix*self.nimgs)
            JJ = np.tile(np.arange(self.npix), self.nimgs)
            A += coo_matrix((data, (II, JJ)))
            A = A.tocoo()
            M = (A.T)*spdiags(D.T.reshape(-1), 0, self.nimgs*self.npix, self.nimgs*self.npix)*A
            rhs = chi*rho_rep*(-rho_rep*psi+self.I.reshape(self.npix, self.nimgs))*w
            rhs = np.mat((A.T)*rhs.T.reshape(-1)).T
            if(it % 5 == 1):
                precond = splu(M.tocsc())
            z_tilde = np.array(z_tilde+precond.solve(rhs).T).reshape(-1)

            # Auxiliary variables
            self.z[imask] = np.exp(z_tilde)
            zx = (np.array(z_tilde)*(Dx.T)).reshape(-1)
            zy = (np.array(z_tilde)*(Dy.T)).reshape(-1)
            Nx = np.zeros((self.nrows, self.ncols))
            Ny = np.zeros((self.nrows, self.ncols))
            Nz = np.zeros((self.nrows, self.ncols))
            Nx[imask] = self.fx*zx
            Ny[imask] = self.fy*zy
            Nz[imask] = -u_tilde[imask]*zx-v_tilde[imask]*zy-1
            dz = np.sqrt(Nx*Nx+Ny*Ny+Nz*Nz)
            X = np.array(self.z[imask] * u_tilde[imask]/self.fx)
            Y = np.array(self.z[imask] * v_tilde[imask]/self.fy)
            Z = np.array(self.z[imask])

            # Convergence test
            energy_new = 0
            [Tz, grad_Tz] = self.t_fcn(
                z_tilde, u_tilde[imask]/self.fx, v_tilde[imask]/self.fy)
            psi = self.shading_fcn(z_tilde, Tz, px_rep, py_rep, Dx_rep, Dy_rep)
            psi = psi.reshape((self.nimgs, self.npix)).T
            energy_new += self.J_fcn(rho_tilde, np.multiply(W_idx,psi), np.multiply(W_idx, self.I))
            relative_diff = abs(energy_new-energy)/energy_new
            energy = energy_new
            print(it)
            print(energy)
            print(relative_diff)

            if(relative_diff < self.tol):
                break
        return X, Y, Z, N

    def make_gradient(self):
        Dyp, Dym, Dxp, Dxm, Omega, index_matrix = self.graddient_operators()

        Dy = Dyp
        no_bottom = np.where(~Omega[0])
        no_bottom = index_matrix[no_bottom][np.nonzero(
            index_matrix[no_bottom])].astype('int32') - 1
        Dy[no_bottom, :] = Dym[no_bottom, :]

        Dx = Dxp
        no_right = np.where(~Omega[2])
        no_right = index_matrix[no_right][np.nonzero(
            index_matrix[no_right])].astype('int32') - 1
        Dx[no_right, :] = Dxm[no_right, :]

        return Dx, Dy

    def graddient_operators(self):
        self.nrows = self.mask.shape[0]
        self.ncols = self.mask.shape[1]
        Omega_padded = np.pad(self.mask, 1, 'constant')

        Omega = [0] * 4
        # Pixels who have bottom neighbor in mask
        Omega[0] = np.multiply(self.mask, Omega_padded[2:, 1:-1]).astype(bool)
        # Pixels who have top neighbor in mask
        Omega[1] = np.multiply(self.mask, Omega_padded[:-2, 1:-1]).astype(bool)
        # Pixels who have right neighbor in mask
        Omega[2] = np.multiply(self.mask, Omega_padded[1:-1, 2:]).astype(bool)
        # Pixels who have left neighbor in mask
        Omega[3] = np.multiply(self.mask, Omega_padded[1:-1, :-2]).astype(bool)

        imask = np.where(self.mask.T > 0)
        index_matrix = np.zeros([self.nrows, self.ncols]).T
        index_matrix[imask] = range(1, 1 + len(imask[0]))
        index_matrix = index_matrix.T

        # Dv matrix
        # When there is a neighbor on the right : forward differences
        idx_c = np.where(Omega[2] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0], idx_c[1] + 1)].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = np.ones(len(indices_centre))
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack((KK, -KK))

        Dvp = coo_matrix((KK, (II, JJ)), shape=(
            len(imask[0]), len(imask[0]))).tolil()

        # When there is a neighbor on the left : backword differences
        idx_c = np.where(Omega[3] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0], idx_c[1] - 1)].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = -np.ones([len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack((KK, -KK))

        Dvm = coo_matrix((KK, (II, JJ)), shape=(
            len(imask[0]), len(imask[0]))).tolil()

        # When there is a neighbor on the bottom : forward differences
        idx_c = np.where(Omega[0] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0] + 1, idx_c[1])].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = np.ones([len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack([KK, -KK])

        Dup = coo_matrix((KK, (II, JJ)), shape=(
            len(imask[0]), len(imask[0]))).tolil()

        # When there is a neighbor on the top : backword differences
        idx_c = np.where(Omega[1] > 0)
        indices_centre = index_matrix[idx_c].astype('int32') - 1
        indices_right = index_matrix[(
            idx_c[0] - 1, idx_c[1])].astype('int32') - 1
        II = indices_centre
        JJ = indices_right
        KK = -np.ones([len(indices_centre)])
        II = np.hstack((II, indices_centre))
        JJ = np.hstack((JJ, indices_centre))
        KK = np.hstack([KK, -KK])

        Dum = coo_matrix((KK, (II, JJ)), shape=(
            len(imask[0]), len(imask[0]))).tolil()

        return Dup, Dum, Dvp, Dvm, Omega, index_matrix

    def sort_linear_index(self, sortDim, sortOrder):
        W_idx = np.zeros(self.I.shape)
        indices = np.array([1, 2, 3, 4, 5])
        W_idx[:, indices] = 1

        sortIndex = (-self.I).argsort(1)
        sortindexq = np.unravel_index(sortIndex, self.I.shape)
        return W_idx[sortindexq]

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
            return (abs(x) <= self.lambda_) * (1 - pow(1 - x * x / (self.lambda_ * self.lambda_), 3)) * self.lambda_ * self.lambda_ + (abs(x) < self.lambda_) * (self.lambda_ * self.lambda_)

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
            return np.fmax(x, 0)
        else:
            x

    def chi_fcn(self, x):
        if(self.shadows):
            return (x >= 0) * np.ones(x.shape)
        else:
            return 1 * np.ones(x.shape)

    def shading_fcn(self, z, tz, px_rep, py_rep, Dx_rep, Dy_rep):
        spdiagsx = (self.fx * tz[0, :, :] - px_rep * tz[2, :, :]).T.reshape(-1)
        spdiagsx = spdiags(spdiagsx, 0, self.npix *
                           self.nimgs, self.npix * self.nimgs)
        resx = spdiagsx * Dx_rep

        spdiagsy = (self.fy * tz[1, :, :] - py_rep * tz[2, :, :]).T.reshape(-1)
        spdiagsy = spdiags(spdiagsy, 0, self.npix *
                           self.nimgs, self.npix * self.nimgs)
        resy = spdiagsy * Dy_rep

        resz = tz[2, :, :].T.reshape(-1)

        res = (resx + resy) * z - resz
        return res

    def r_fcn(self, rho, shadz, II):
        II = II.T.reshape(-1)
        shadz = shadz.T.reshape(-1)
        res = np.kron(np.ones((1,self.nimgs)),rho) * self.psi_fcn(shadz) - II
        res = res.reshape(-1)
        return res

    def J_fcn(self, rho, shadz, II):
        res = self.r_fcn(rho, shadz, II)
        res = self.phi_fcn(res)
        res = sum(res)
        return res

    def t_fcn(self, z, u_tilde, v_tilde):
        self.npix = len(z)
        # Current mesh
        exp_z = np.exp(z)
        X = exp_z * u_tilde
        Y = exp_z * v_tilde
        Z = exp_z
        # T_field
        T_field = np.zeros((3, self.npix, self.nimgs))
        a_field = np.zeros((self.npix, self.nimgs))
        da_field = np.zeros((self.npix, self.nimgs))
        for i in range(self.nimgs):
            # unit lighting filed
            T_field[0, :, i] = self.S[i, 0] - X
            T_field[1, :, i] = self.S[i, 1] - Y
            T_field[2, :, i] = self.S[i, 2] - Z
            normS_i = np.sqrt(pow(T_field[0, :, i], 2) +
                              pow(T_field[1, :, i], 2) +
                              pow(T_field[2, :, i], 2))
            # attenuation = anisotropy / squared distance
            scal_prod = -T_field[0, :, i] * self.Dir[i, 0] - \
                T_field[1, :, i] * self.Dir[i, 1] - \
                T_field[2, :, i] * self.Dir[i, 2]
            a_field[:, i] = pow(scal_prod, self.mu[i]) / \
                pow(normS_i, (3 + self.mu[i]))
            da_field[:, i] = self.mu[i] * pow(scal_prod, self.mu[i] - 1) * (X* self.Dir[i, 0] + Y * self.Dir[i, 1] + Z * self.Dir[i, 2]) / (pow(normS_i, self.mu[i] + 3)) - (
                self.mu[i] + 3) * pow(scal_prod, self.mu[i]) * (-T_field[0, :, i] * X - T_field[1, :, i] * Y - T_field[2, :, i] * Z) / pow(normS_i, self.mu[i] + 5)
            # Final lighting field
            T_field[0, :, i] = T_field[0, :, i] * a_field[:, i]
            T_field[1, :, i] = T_field[1, :, i] * a_field[:, i]
            T_field[2, :, i] = T_field[2, :, i] * a_field[:, i]

        grad_t = np.zeros((3, self.npix, self.nimgs))
        grad_t[0] = ((-exp_z * u_tilde).T * (a_field +
                                             da_field).T).T + (self.S[:, 0] * da_field)
        grad_t[1] = ((-exp_z * v_tilde).T * (a_field +
                                             da_field).T).T + (self.S[:, 1] * da_field)
        grad_t[2] = (-exp_z.T * (a_field + da_field).T).T + \
            (self.S[:, 2] * da_field)

        return T_field, grad_t