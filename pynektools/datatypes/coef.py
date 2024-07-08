import numpy as np
from math import pi


class coef_c:
    def __init__(self, msh, comm):

        self.gdim = msh.gdim

        self.V, self.Vinv, self.w3, self.x, self.w = get_transform_matrix(
            msh.lx, msh.gdim
        )

        self.dr, self.ds, self.dt, self.dn = get_derivative_matrix(msh.lx, msh.gdim)

        # Find the components of the jacobian per point
        # jac(x,y,z) = [dxdr, dxds, dxdt ; dydr, dyds, dydt; dzdr, dzds, dzdt]
        self.dxdr = self.dudrst(msh.x, self.dr)
        self.dxds = self.dudrst(msh.x, self.ds)
        if msh.gdim > 2:
            self.dxdt = self.dudrst(msh.x, self.dt)

        self.dydr = self.dudrst(msh.y, self.dr)
        self.dyds = self.dudrst(msh.y, self.ds)
        if msh.gdim > 2:
            self.dydt = self.dudrst(msh.y, self.dt)

        if msh.gdim > 2:
            self.dzdr = self.dudrst(msh.z, self.dr)
            self.dzds = self.dudrst(msh.z, self.ds)
            self.dzdt = self.dudrst(msh.z, self.dt)

        self.drdx = np.zeros_like(self.dxdr, dtype=np.double)
        self.drdy = np.zeros_like(self.dxdr, dtype=np.double)
        if msh.gdim > 2:
            self.drdz = np.zeros_like(self.dxdr, dtype=np.double)

        self.dsdx = np.zeros_like(self.dxdr, dtype=np.double)
        self.dsdy = np.zeros_like(self.dxdr, dtype=np.double)
        if msh.gdim > 2:
            self.dsdz = np.zeros_like(self.dxdr, dtype=np.double)

        if msh.gdim > 2:
            self.dtdx = np.zeros_like(self.dxdr, dtype=np.double)
            self.dtdy = np.zeros_like(self.dxdr, dtype=np.double)
            self.dtdz = np.zeros_like(self.dxdr, dtype=np.double)

        # Find the jacobian determinant, its inverse inverse and mass matrix (3D)
        # This maps dxyz/drst
        # Gere we store the jacobian determinant as "jac"
        self.jac = np.zeros_like(self.dxdr, dtype=np.double)
        # This maps drst/dxyz
        self.jac_inv = np.zeros_like(self.dxdr, dtype=np.double)
        self.B = np.zeros_like(self.dxdr, dtype=np.double)

        if msh.gdim > 2:
            temp_mat = np.zeros((3, 3))
        else:
            temp_mat = np.zeros((2, 2))

        for e in range(0, msh.nelv):
            for k in range(0, msh.lz):
                for j in range(0, msh.ly):
                    for i in range(0, msh.lx):
                        temp_mat[0, 0] = self.dxdr[e, k, j, i]
                        temp_mat[0, 1] = self.dxds[e, k, j, i]
                        if msh.gdim > 2:
                            temp_mat[0, 2] = self.dxdt[e, k, j, i]

                        temp_mat[1, 0] = self.dydr[e, k, j, i]
                        temp_mat[1, 1] = self.dyds[e, k, j, i]
                        if msh.gdim > 2:
                            temp_mat[1, 2] = self.dydt[e, k, j, i]

                        if msh.gdim > 2:
                            temp_mat[2, 0] = self.dzdr[e, k, j, i]
                            temp_mat[2, 1] = self.dzds[e, k, j, i]
                            temp_mat[2, 2] = self.dzdt[e, k, j, i]

                        # Fill the jaconian determinant, its inverse and the mass matrix
                        self.jac[e, k, j, i] = np.linalg.det(temp_mat)
                        self.jac_inv[e, k, j, i] = 1 / self.jac[e, k, j, i]
                        self.B[e, k, j, i] = (
                            self.jac[e, k, j, i]
                            * np.diag(self.w3).reshape((msh.lz, msh.ly, msh.lx))[
                                k, j, i
                            ]
                        )
                        # Fill the terms for the jacobian inverse

                        # Find the components of the inverse of the jacobian per point
                        temp_mat_inv = np.linalg.inv(
                            temp_mat
                        )  # Note that here, 1/det(jac) is already performed

                        # Find the components of the jacobian per point
                        # jac_inv(r,s,t) = [drdx, drdy, drdz ; dsdx, dsdy, dsdz; dtdx, dtdy, dtdz]
                        self.drdx[e, k, j, i] = temp_mat_inv[0, 0]
                        self.drdy[e, k, j, i] = temp_mat_inv[0, 1]
                        if msh.gdim > 2:
                            self.drdz[e, k, j, i] = temp_mat_inv[0, 2]

                        self.dsdx[e, k, j, i] = temp_mat_inv[1, 0]
                        self.dsdy[e, k, j, i] = temp_mat_inv[1, 1]
                        if msh.gdim > 2:
                            self.dsdz[e, k, j, i] = temp_mat_inv[1, 2]

                        if msh.gdim > 2:
                            self.dtdx[e, k, j, i] = temp_mat_inv[2, 0]
                            self.dtdy[e, k, j, i] = temp_mat_inv[2, 1]
                            self.dtdz[e, k, j, i] = temp_mat_inv[2, 2]

        # Get area stuff only if mesh is 3D
        # Remember that the area described by two vectors is given by the norm of its cross product
        # i.e., norm(drxds)
        # Here we do that and then multiply by the weights.
        # Similar to what we do with the volume mass matrix. Where we calculate the jacobian determinant
        # and then multiply with weights
        if msh.gdim > 2:
            d1 = np.zeros((3), dtype=np.double)
            d2 = np.zeros((3), dtype=np.double)
            self.area = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)
            self.nx = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)
            self.ny = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)
            self.nz = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=np.double)

            # ds x dt
            for e in range(0, msh.nelv):
                for k in range(0, msh.lz):
                    for j in range(0, msh.ly):
                        weight = self.w[j] * self.w[k]

                        # For facet 1
                        d1[0] = self.dxds[e, k, j, 0]
                        d1[1] = self.dyds[e, k, j, 0]
                        d1[2] = self.dzds[e, k, j, 0]
                        d2[0] = self.dxdt[e, k, j, 0]
                        d2[1] = self.dydt[e, k, j, 0]
                        d2[2] = self.dzdt[e, k, j, 0]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 0, k, j] = norm * weight
                        self.nx[e, 0, k, j] = -cross[0] / norm
                        self.ny[e, 0, k, j] = -cross[1] / norm
                        self.nz[e, 0, k, j] = -cross[2] / norm

                        # For facet 2
                        d1[0] = self.dxds[e, k, j, msh.lx - 1]
                        d1[1] = self.dyds[e, k, j, msh.lx - 1]
                        d1[2] = self.dzds[e, k, j, msh.lx - 1]
                        d2[0] = self.dxdt[e, k, j, msh.lx - 1]
                        d2[1] = self.dydt[e, k, j, msh.lx - 1]
                        d2[2] = self.dzdt[e, k, j, msh.lx - 1]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 1, k, j] = norm * weight
                        self.nx[e, 1, k, j] = cross[0] / norm
                        self.ny[e, 1, k, j] = cross[1] / norm
                        self.nz[e, 1, k, j] = cross[2] / norm

            # dr x dt
            for e in range(0, msh.nelv):
                for k in range(0, msh.lz):
                    for i in range(0, msh.lx):
                        weight = self.w[i] * self.w[k]

                        # For facet 3
                        d1[0] = self.dxdr[e, k, 0, i]
                        d1[1] = self.dydr[e, k, 0, i]
                        d1[2] = self.dzdr[e, k, 0, i]
                        d2[0] = self.dxdt[e, k, 0, i]
                        d2[1] = self.dydt[e, k, 0, i]
                        d2[2] = self.dzdt[e, k, 0, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 2, k, i] = norm * weight
                        self.nx[e, 2, k, i] = cross[0] / norm
                        self.ny[e, 2, k, i] = cross[1] / norm
                        self.nz[e, 2, k, i] = cross[2] / norm

                        # For facet 4
                        d1[0] = self.dxdr[e, k, msh.ly - 1, i]
                        d1[1] = self.dydr[e, k, msh.ly - 1, i]
                        d1[2] = self.dzdr[e, k, msh.ly - 1, i]
                        d2[0] = self.dxdt[e, k, msh.ly - 1, i]
                        d2[1] = self.dydt[e, k, msh.ly - 1, i]
                        d2[2] = self.dzdt[e, k, msh.ly - 1, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 3, k, i] = norm * weight
                        self.nx[e, 3, k, i] = -cross[0] / norm
                        self.ny[e, 3, k, i] = -cross[1] / norm
                        self.nz[e, 3, k, i] = -cross[2] / norm

            # dr x ds
            for e in range(0, msh.nelv):
                for j in range(0, msh.ly):
                    for i in range(0, msh.lx):
                        weight = self.w[j] * self.w[i]

                        # For facet 5
                        d1[0] = self.dxdr[e, 0, j, i]
                        d1[1] = self.dydr[e, 0, j, i]
                        d1[2] = self.dzdr[e, 0, j, i]
                        d2[0] = self.dxds[e, 0, j, i]
                        d2[1] = self.dyds[e, 0, j, i]
                        d2[2] = self.dzds[e, 0, j, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 4, j, i] = norm * weight
                        self.nx[e, 4, j, i] = -cross[0] / norm
                        self.ny[e, 4, j, i] = -cross[1] / norm
                        self.nz[e, 4, j, i] = -cross[2] / norm

                        # For facet 6
                        d1[0] = self.dxdr[e, msh.lz - 1, j, i]
                        d1[1] = self.dydr[e, msh.lz - 1, j, i]
                        d1[2] = self.dzdr[e, msh.lz - 1, j, i]
                        d2[0] = self.dxds[e, msh.lz - 1, j, i]
                        d2[1] = self.dyds[e, msh.lz - 1, j, i]
                        d2[2] = self.dzds[e, msh.lz - 1, j, i]
                        cross = np.cross(d1, d2)
                        norm = np.linalg.norm(cross)
                        self.area[e, 5, j, i] = norm * weight
                        self.nx[e, 5, j, i] = cross[0] / norm
                        self.ny[e, 5, j, i] = cross[1] / norm
                        self.nz[e, 5, j, i] = cross[2] / norm

        return

    def dudrst(self, field, dr):

        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        dudrst = np.zeros_like(field, dtype=np.double)

        for e in range(0, nelv):
            tmp = field[e, :, :, :].reshape(-1, 1)
            dtmp = dr @ tmp
            dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))

        return dudrst

    def dudxyz(self, field, drdx, dsdx, dtdx=None):

        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]
        dudxyz = np.zeros_like(field, dtype=np.double)

        dfdr = self.dudrst(field, self.dr)
        dfds = self.dudrst(field, self.ds)
        if self.gdim > 2:
            dfdt = self.dudrst(field, self.dt)

        # NOTE: DO NOT NEED TO MULTIPLY BY INVERSE OF JACOBIAN DETERMINAT.
        # THIS STEP IS ALREADY DONE IF YOU CALCULATED THE INVERSE WITH NUMPY
        # Here we multiply the derivative in reference element with the respective
        # row of the jacobian
        if self.gdim > 2:
            for e in range(0, nelv):
                for k in range(0, lz):
                    for j in range(0, ly):
                        for i in range(0, lx):
                            # dudxyz[e,k,j,i] = self.jac_inv[e, k, j, i] * (dfdr[e, k, j, i] *drdx[e, k, j, i]  + dfds[e, k, j, i] *dsdx[e, k, j, i]  + dfdt[e, k, j, i] *dtdx[e, k, j, i] )
                            dudxyz[e, k, j, i] = (
                                dfdr[e, k, j, i] * drdx[e, k, j, i]
                                + dfds[e, k, j, i] * dsdx[e, k, j, i]
                                + dfdt[e, k, j, i] * dtdx[e, k, j, i]
                            )
        else:
            for e in range(0, nelv):
                for k in range(0, lz):
                    for j in range(0, ly):
                        for i in range(0, lx):
                            # dudxyz[e,k,j,i] = self.jac_inv[e, k, j, i] * ( dfdr[e, k, j, i] * drdx[e, k, j, i] + dfds[e, k, j, i] * dsdx[e, k, j, i] )
                            dudxyz[e, k, j, i] = (
                                dfdr[e, k, j, i] * drdx[e, k, j, i]
                                + dfds[e, k, j, i] * dsdx[e, k, j, i]
                            )

        return dudxyz

    def glsum(self, a, comm, datype=np.double):

        sendbuf = np.ones((1), datype)
        sendbuf[0] = np.sum(a)
        recvbuf = np.zeros((1), datype)
        comm.Allreduce(sendbuf, recvbuf)

        return recvbuf[0]

    def dssum(self, field, msh):
        tmp = np.copy(field)
        for ind in range(0, len(msh.nonlinear_shared_points)):
            ind1 = np.array(msh.nonlinear_indices[ind])
            ind2 = np.array(msh.nonlinear_shared_points[ind])
            field[ind1[:, 0], ind1[:, 1], ind1[:, 2], ind1[:, 3]] = np.mean(
                tmp[ind2[:, 0], ind2[:, 1], ind2[:, 2], ind2[:, 3]]
            )

        return field


# -----------------------------------------------------------------------


def dr_cross_ds(dxdr, dydr, dzdr, dxds, dyds, dzds, msh):

    dr = np.zeros((3), dtype=np.double)
    ds = np.zeros((3), dtype=np.double)

    a = np.zeros_like(dxdr, dtype=np.double)
    b = np.zeros_like(dxdr, dtype=np.double)
    c = np.zeros_like(dxdr, dtype=np.double)
    dot = np.zeros_like(dxdr, dtype=np.double)

    for e in range(0, msh.nelv):
        for k in range(0, msh.lz):
            for j in range(0, msh.ly):
                for i in range(0, msh.lx):

                    dr[0] = dxdr[e, k, j, i]
                    dr[1] = dydr[e, k, j, i]
                    dr[2] = dzdr[e, k, j, i]

                    ds[0] = dxds[e, k, j, i]
                    ds[1] = dyds[e, k, j, i]
                    ds[2] = dzds[e, k, j, i]

                    tmp = np.cross(dr, ds)

                    a[e, k, j, i] = tmp[0]
                    b[e, k, j, i] = tmp[1]
                    c[e, k, j, i] = tmp[2]
                    dot[e, k, j, i] = np.dot(tmp, tmp)

    return a, b, c, dot


## Define functions for the calculation of the quadrature points (Taken from the lecture notes)
def GLC_pwts(n):
    """
    Gauss-Lobatto-Chebyshev (GLC) points and weights over [-1,1]
    Args:
      `n`: int, number of nodes
    Returns
       `x`: 1D numpy array of size `n`, nodes
       `w`: 1D numpy array of size `n`, weights
    """

    def delt(i, n):
        del_ = 1.0
        if i == 0 or i == n - 1:
            del_ = 0.5
        return del_

    x = np.cos(np.arange(n) * pi / (n - 1))
    w = np.zeros(n)
    for i in range(n):
        tmp_ = 0.0
        for k in range(int((n - 1) / 2)):
            tmp_ += delt(2 * k, n) / (1 - 4.0 * k**2) * np.cos(2 * i * pi * k / (n - 1))
        w[i] = tmp_ * delt(i, n) * 4 / float(n - 1)
    return x, w


def GLL_pwts(n, eps=10**-8, maxIter=1000):
    """
    Generating `n `Gauss-Lobatto-Legendre (GLL) nodes and weights using the
    Newton-Raphson iteration.
    Args:
      `n`: int
         Number of GLL nodes
      `eps`: float (optional)
         Min error to keep the iteration running
      `maxIter`: float (optional)
         Max number of iterations
    Outputs:
      `xi`: 1D numpy array of size `n`
         GLL nodes
      `w`: 1D numpy array of size `n`
         GLL weights
    Reference:
       Canuto C., Hussaini M. Y., Quarteroni A., Tang T. A.,
       "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
       https://link.springer.com/book/10.1007/978-3-642-84108-8
    """
    V = np.zeros((n, n))  # Legendre Vandermonde Matrix
    # Initial guess for the nodes: GLC points
    xi, w_ = GLC_pwts(n)
    iter_ = 0
    err = 1000
    xi_old = xi
    while iter_ < maxIter and err > eps:
        iter_ += 1
        # Update the Legendre-Vandermonde matrix
        V[:, 0] = 1.0
        V[:, 1] = xi
        for j in range(2, n):
            V[:, j] = (
                (2.0 * j - 1) * xi * V[:, j - 1] - (j - 1) * V[:, j - 2]
            ) / float(j)
        # Newton-Raphson iteration
        xi = xi_old - (xi * V[:, n - 1] - V[:, n - 2]) / (n * V[:, n - 1])
        err = max(abs(xi - xi_old).flatten())
        xi_old = xi
    if iter_ > maxIter and err > eps:
        print("gllPts(): max iterations reached without convergence!")
    # Weights
    w = 2.0 / (n * (n - 1) * V[:, n - 1] ** 2.0)
    return xi, w


def get_transform_matrix(n, dim):
    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    WW = np.eye(n)
    for i in range(0, n):
        WW[i, i] = w[i]

    ## First we need the legendre polynomials
    # order of the polynomials
    p = n
    # Create a counter for the loops
    p_v = np.arange(p)

    # get the legendre polynomial matrix

    # The polynomials are stored in a matrix with the following structure:
    #  |  pi_0(x0)  ... pi_0(x_n)
    #  |  pi_1(x0)  ... pi_1(x_n)
    #  |  ...
    #  |  pi_p(x0)  ... pi_p(x_n)
    #  The acending rows represent accending polynomial order, the different columns represent different x_i

    # Allocate space
    Leg = np.zeros((p, p))
    # First row is filled with 1 according to recursive formula
    Leg[0, :] = np.ones((1, p))
    # Second row is filled with x according to recursive formula
    Leg[1, :] = np.multiply(np.ones((1, p)), x)

    # Apply the recursive formula for all x_i
    for j in range(1, len(p_v) - 1):
        for k_ in p_v:
            Leg[j + 1, k_] = ((2 * j + 1) * x[k_] * Leg[j, k_] - j * Leg[j - 1, k_]) / (
                j + 1
            )

    Leg = Leg.T  # nek and I transpose it for transform

    # Scaling factor as in books
    delta = np.ones(n)
    for i in range(0, n):
        # delta[i]=2/(2*i+1)       #it is the same both ways
        delta[i] = 2 / (2 * (i + 1) - 1)
    delta[n - 1] = 2 / (n - 1)
    # print(delta)
    # Scaling factor to normalize
    for i in range(0, n):
        delta[i] = np.sqrt(1 / delta[i])

    # apply the scaling factor
    for i in range(0, n):
        for j in range(0, n):
            Leg[i, j] = Leg[i, j] * delta[j]

    AA = np.matmul(Leg.T, np.matmul(WW, Leg))

    # 2d transformation matrix
    v = Leg
    V2d = np.kron(v, v)
    vinv = Leg.T @ WW
    Vinv2d = np.kron(vinv, vinv)

    # 3d transformation matrix
    v = Leg
    V = np.kron(v, np.kron(v, v))
    vinv = Leg.T @ WW
    Vinv = np.kron(vinv, np.kron(vinv, vinv))

    if dim == 1:
        V = v
        Vinv = vinv
    if dim == 2:
        V = V2d
        Vinv = Vinv2d
        W3 = np.kron(WW, WW)
    if dim == 3:
        V = V
        Vinv = Vinv
        W3 = np.kron(WW, np.kron(WW, WW))

    return V, Vinv, W3, x, w


def get_derivative_matrix(n, dim):
    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    WW = np.eye(n)
    for i in range(0, n):
        WW[i, i] = w[i]

    ## First we need the legendre polynomials
    # order of the polynomials
    p = n
    # Create a counter for the loops
    p_v = np.arange(p)

    # get the legendre polynomial matrix

    # The polynomials are stored in a matrix with the following structure:
    #  |  pi_0(x0)  ... pi_0(x_n)
    #  |  pi_1(x0)  ... pi_1(x_n)
    #  |  ...
    #  |  pi_p(x0)  ... pi_p(x_n)
    #  The acending rows represent accending polynomial order, the different columns represent different x_i

    # Allocate space
    Leg = np.zeros((p, p))
    # First row is filled with 1 according to recursive formula
    Leg[0, :] = np.ones((1, p))
    # Second row is filled with x according to recursive formula
    Leg[1, :] = np.multiply(np.ones((1, p)), x)

    # Apply the recursive formula for all x_i
    for j in range(1, len(p_v) - 1):
        for k_ in p_v:
            Leg[j + 1, k_] = ((2 * j + 1) * x[k_] * Leg[j, k_] - j * Leg[j - 1, k_]) / (
                j + 1
            )

    D_N = np.zeros((p, p))

    # Simply apply the values as given in the book
    for i in range(0, len(p_v)):
        for j in range(0, len(p_v)):
            if i != j:
                D_N[i, j] = (Leg[p - 1, i] / Leg[p - 1, j]) * (1 / (x[i] - x[j]))
            if i == 0 and j == 0:
                D_N[i, j] = -(((p - 1) + 1) * (p - 1)) / 4
            if i == (p - 1) and j == (p - 1):
                D_N[i, j] = (((p - 1) + 1) * (p - 1)) / 4
            if i == j and i != 0 and i != (p - 1):
                D_N[i, j] = 0

    if dim == 1:
        DX = D_N
        DY = None
        DZ = None

    if dim == 2:
        DX2D = np.kron(np.eye(p), D_N)
        DY2D = np.kron(D_N, np.eye(p))

        DX = DX2D
        DY = DY2D
        DZ = None

    if dim == 3:
        DX3D = np.kron(np.eye(p), np.kron(np.eye(p), D_N))
        DY3D = np.kron(np.eye(p), np.kron(D_N, np.eye(p)))
        DZ3D = np.kron(D_N, np.kron(np.eye(p), np.eye(p)))

        DX = DX3D
        DY = DY3D
        DZ = DZ3D

    return DX, DY, DZ, D_N
