import numpy as np
from math import pi
from scipy.special import legendre


class element_interpolator_c:
    def __init__(self, n, modal_search=True):

        # Order of the element
        self.n = n

        # Get reference element
        get_reference_element(self)

        # Get reference element
        get_legendre_transformation_matrices(self)

        # Allocate relevant arrays
        self.xj = np.zeros(1)
        self.yj = np.zeros(1)
        self.zj = np.zeros(1)
        self.rj = np.zeros_like(self.xj)
        self.sj = np.zeros_like(self.yj)
        self.tj = np.zeros_like(self.zj)
        self.rstj = np.zeros((3, 1))
        self.eps_rst = np.ones((3, 1))
        self.jac = np.zeros((3, 3))
        self.iterations = 0
        self.field_e = np.zeros_like(self.x_e)
        self.point_inside_element = False

        self.modal_search = modal_search

        return

    def project_element_into_basis(self, x_e, y_e, z_e, apply_1d_ops=True):

        # Assing the inputs to proper formats
        self.x_e[:, 0] = x_e[:, :, :].reshape(-1, 1)[:, 0]
        self.y_e[:, 0] = y_e[:, :, :].reshape(-1, 1)[:, 0]
        self.z_e[:, 0] = z_e[:, :, :].reshape(-1, 1)[:, 0]

        if self.modal_search:
            # Get the modal representation
            if not apply_1d_ops:
                self.x_e_hat = self.v_inv @ self.x_e
                self.y_e_hat = self.v_inv @ self.y_e
                self.z_e_hat = self.v_inv @ self.z_e
            elif apply_1d_ops:
                self.x_e_hat = apply_operators_3d(
                    self.v1d_inv, self.v1d_inv, self.v1d_inv, self.x_e
                )
                self.y_e_hat = apply_operators_3d(
                    self.v1d_inv, self.v1d_inv, self.v1d_inv, self.y_e
                )
                self.z_e_hat = apply_operators_3d(
                    self.v1d_inv, self.v1d_inv, self.v1d_inv, self.z_e
                )
        else:
            # Use the nodal representation
            self.x_e_hat = self.x_e
            self.y_e_hat = self.y_e
            self.z_e_hat = self.z_e

        return

    def find_rst_from_xyz(
        self, xj, yj, zj, tol=np.finfo(np.double).eps * 10, max_iterations=1000
    ):

        self.point_inside_element = False

        self.xj[0] = xj
        self.yj[0] = yj
        self.zj[0] = zj

        # Determine the initial conditions
        determine_initial_guess(self)

        # Use the newton method to identify the coordinates
        self.iterations = 0
        self.eps_rst[:, :] = 1
        while np.linalg.norm(self.eps_rst) > tol and self.iterations < max_iterations:

            # Update the guess
            self.rstj[0, 0] = self.rj[0]
            self.rstj[1, 0] = self.sj[0]
            self.rstj[2, 0] = self.tj[0]

            # Estimate the xyz values from rst and also the jacobian (it is updated inside self.jac)
            xj_found, yj_found, zj_found = self.get_xyz_from_rst(
                self.rj[0], self.sj[0], self.tj[0]
            )

            # Find the residuals and the jacobian inverse.
            self.eps_rst[0, 0] = self.xj[0] - xj_found
            self.eps_rst[1, 0] = self.yj[0] - yj_found
            self.eps_rst[2, 0] = self.zj[0] - zj_found
            jac_inv = np.linalg.inv(self.jac)

            # Find the new guess
            self.rstj = self.rstj - (0 - jac_inv @ self.eps_rst)

            # Update the values
            self.rj[0] = self.rstj[0, 0]
            self.sj[0] = self.rstj[1, 0]
            self.tj[0] = self.rstj[2, 0]
            self.iterations += 1

        limit = 1 + np.finfo(np.single).eps
        if (
            abs(self.rj[0]) <= limit
            and abs(self.sj[0]) <= limit
            and abs(self.tj[0]) <= limit
        ):
            self.point_inside_element = True
        else:
            self.point_inside_element = False

        return self.rj[0], self.sj[0], self.tj[0]

    def get_xyz_from_rst(self, rj, sj, tj, apply_1d_ops=True):

        n = self.n
        self.rj[0] = rj
        self.sj[0] = sj
        self.tj[0] = tj

        # Find the basis for each coordinate separately
        if self.modal_search:
            # If modal search, the basis is legendre
            ortho_basis_rj = legendre_basis_at_xtest(n, self.rj)
            ortho_basis_sj = legendre_basis_at_xtest(n, self.sj)
            ortho_basis_tj = legendre_basis_at_xtest(n, self.tj)

            ortho_basis_prm_rj = legendre_basis_derivative_at_xtest(
                ortho_basis_rj, self.rj
            )
            ortho_basis_prm_sj = legendre_basis_derivative_at_xtest(
                ortho_basis_sj, self.sj
            )
            ortho_basis_prm_tj = legendre_basis_derivative_at_xtest(
                ortho_basis_tj, self.tj
            )
        else:
            # If nodal search, the basis is lagrange
            ortho_basis_rj = lagInterp_matrix_at_xtest(self.x_gll, self.rj)
            ortho_basis_sj = lagInterp_matrix_at_xtest(self.x_gll, self.sj)
            ortho_basis_tj = lagInterp_matrix_at_xtest(self.x_gll, self.tj)

            ortho_basis_prm_rj = lagInterp_derivative_matrix_at_xtest(
                self.x_gll, ortho_basis_rj, self.rj
            )
            ortho_basis_prm_sj = lagInterp_derivative_matrix_at_xtest(
                self.x_gll, ortho_basis_sj, self.sj
            )
            ortho_basis_prm_tj = lagInterp_derivative_matrix_at_xtest(
                self.x_gll, ortho_basis_tj, self.tj
            )

        if not apply_1d_ops:
            # Construct the 3d basis
            ortho_basis_rstj = np.kron(
                ortho_basis_tj.T, np.kron(ortho_basis_sj.T, ortho_basis_rj.T)
            )

            ortho_basis_drj = np.kron(
                ortho_basis_tj.T, np.kron(ortho_basis_sj.T, ortho_basis_prm_rj.T)
            )
            ortho_basis_dsj = np.kron(
                ortho_basis_tj.T, np.kron(ortho_basis_prm_sj.T, ortho_basis_rj.T)
            )
            ortho_basis_dtj = np.kron(
                ortho_basis_prm_tj.T, np.kron(ortho_basis_sj.T, ortho_basis_rj.T)
            )

            x = (ortho_basis_rstj @ self.x_e_hat)[0, 0]
            y = (ortho_basis_rstj @ self.y_e_hat)[0, 0]
            z = (ortho_basis_rstj @ self.z_e_hat)[0, 0]

            self.jac[0, 0] = (ortho_basis_drj @ self.x_e_hat)[0, 0]
            self.jac[0, 1] = (ortho_basis_dsj @ self.x_e_hat)[0, 0]
            self.jac[0, 2] = (ortho_basis_dtj @ self.x_e_hat)[0, 0]

            self.jac[1, 0] = (ortho_basis_drj @ self.y_e_hat)[0, 0]
            self.jac[1, 1] = (ortho_basis_dsj @ self.y_e_hat)[0, 0]
            self.jac[1, 2] = (ortho_basis_dtj @ self.y_e_hat)[0, 0]

            self.jac[2, 0] = (ortho_basis_drj @ self.z_e_hat)[0, 0]
            self.jac[2, 1] = (ortho_basis_dsj @ self.z_e_hat)[0, 0]
            self.jac[2, 2] = (ortho_basis_dtj @ self.z_e_hat)[0, 0]

        elif apply_1d_ops:
            # Apply the 1d operators to the 3d field
            x = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.x_e_hat
            )[0, 0]
            y = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.y_e_hat
            )[0, 0]
            z = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.z_e_hat
            )[0, 0]

            self.jac[0, 0] = apply_operators_3d(
                ortho_basis_prm_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.x_e_hat
            )[0, 0]
            self.jac[0, 1] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_prm_sj.T, ortho_basis_tj.T, self.x_e_hat
            )[0, 0]
            self.jac[0, 2] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_prm_tj.T, self.x_e_hat
            )[0, 0]

            self.jac[1, 0] = apply_operators_3d(
                ortho_basis_prm_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.y_e_hat
            )[0, 0]
            self.jac[1, 1] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_prm_sj.T, ortho_basis_tj.T, self.y_e_hat
            )[0, 0]
            self.jac[1, 2] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_prm_tj.T, self.y_e_hat
            )[0, 0]

            self.jac[2, 0] = apply_operators_3d(
                ortho_basis_prm_rj.T, ortho_basis_sj.T, ortho_basis_tj.T, self.z_e_hat
            )[0, 0]
            self.jac[2, 1] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_prm_sj.T, ortho_basis_tj.T, self.z_e_hat
            )[0, 0]
            self.jac[2, 2] = apply_operators_3d(
                ortho_basis_rj.T, ortho_basis_sj.T, ortho_basis_prm_tj.T, self.z_e_hat
            )[0, 0]

        return x, y, z

    def interpolate_field_at_rst(self, rj, sj, tj, field_e, apply_1d_ops=True):

        r_j = np.ones((1))
        s_j = np.ones((1))
        t_j = np.ones((1))

        r_j[0] = rj
        s_j[0] = sj
        t_j[0] = tj

        self.field_e[:, 0] = field_e[:, :, :].reshape(-1, 1)[:, 0]

        lk_r = lagInterp_matrix_at_xtest(self.x_gll, r_j)
        lk_s = lagInterp_matrix_at_xtest(self.x_gll, s_j)
        lk_t = lagInterp_matrix_at_xtest(self.x_gll, t_j)

        if not apply_1d_ops:
            lk_3d = np.kron(lk_t.T, np.kron(lk_s.T, lk_r.T))
            field_at_rst = (lk_3d @ self.field_e)[0, 0]
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(lk_r.T, lk_s.T, lk_t.T, self.field_e)[
                0, 0
            ]

        return field_at_rst

    def interpolate_field_at_rst_vector(self, rj, sj, tj, field_e, apply_1d_ops=True):

        r_j = np.ones((rj.size))
        s_j = np.ones((sj.size))
        t_j = np.ones((tj.size))

        r_j[:] = rj[:]
        s_j[:] = sj[:]
        t_j[:] = tj[:]

        self.field_e[:, 0] = field_e[:, :, :].reshape(-1, 1)[:, 0]

        lk_r = lagInterp_matrix_at_xtest(self.x_gll, r_j)
        lk_s = lagInterp_matrix_at_xtest(self.x_gll, s_j)
        lk_t = lagInterp_matrix_at_xtest(self.x_gll, t_j)

        if not apply_1d_ops:
            lk_3d = np.kron(lk_t.T, np.kron(lk_s.T, lk_r.T))
            field_at_rst = lk_3d @ self.field_e
        elif apply_1d_ops:
            field_at_rst = apply_operators_3d(lk_r.T, lk_s.T, lk_t.T, self.field_e)

        field_at_rst = field_at_rst.reshape((t_j.size, s_j.size, r_j.size))

        return field_at_rst

    # ===========================================================================

    def print_summary(self):
        if self.point_inside_element == True:
            print(
                "the point is inside the element, with (r,s,t): ({},{},{})".format(
                    self.rj[0], self.sj[0], self.tj[0]
                )
            )
        else:
            print("the point is outside the element")
            print("r: {}, abs(r)-1: {} ".format(self.rj[0], abs(self.rj[0]) - 1))
            print("s: {}, abs(s)-1: {} ".format(self.sj[0], abs(self.sj[0]) - 1))
            print("t: {}, abs(t)-1: {} ".format(self.tj[0], abs(self.tj[0]) - 1))

        print(
            "Reached error: {} in {} iterations".format(
                np.linalg.norm(self.eps_rst), self.iterations
            )
        )

        return

    def visualize_data(self, xj_found, yj_found, zj_found, col):

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        n = self.n

        # Creating a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.x_e, self.y_e, self.z_e, c="b", marker="o", alpha=0.2)
        ax.scatter(
            self.xj[0], self.yj[0], self.zj[0], c="k", marker="x", alpha=1, s=300
        )
        ax.scatter(xj_found, yj_found, zj_found, c=col, marker="+", alpha=1, s=300)

        # Plot faces
        ax.plot_surface(
            self.x_e.reshape((n, n, n), order="F")[0, :, :],
            self.y_e.reshape((n, n, n), order="F")[0, :, :],
            self.z_e.reshape((n, n, n), order="F")[0, :, :],
            color="b",
            alpha=0.5,
        )
        ax.plot_surface(
            self.x_e.reshape((n, n, n), order="F")[n - 1, :, :],
            self.y_e.reshape((n, n, n), order="F")[n - 1, :, :],
            self.z_e.reshape((n, n, n), order="F")[n - 1, :, :],
            color="b",
            alpha=0.5,
        )
        ax.plot_surface(
            self.x_e.reshape((n, n, n), order="F")[:, 0, :],
            self.y_e.reshape((n, n, n), order="F")[:, 0, :],
            self.z_e.reshape((n, n, n), order="F")[:, 0, :],
            color="b",
            alpha=0.5,
        )
        ax.plot_surface(
            self.x_e.reshape((n, n, n), order="F")[:, n - 1, :],
            self.y_e.reshape((n, n, n), order="F")[:, n - 1, :],
            self.z_e.reshape((n, n, n), order="F")[:, n - 1, :],
            color="b",
            alpha=0.5,
        )
        ax.plot_surface(
            self.x_e.reshape((n, n, n), order="F")[:, :, 0],
            self.y_e.reshape((n, n, n), order="F")[:, :, 0],
            self.z_e.reshape((n, n, n), order="F")[:, :, 0],
            color="b",
            alpha=0.5,
        )
        ax.plot_surface(
            self.x_e.reshape((n, n, n), order="F")[:, :, n - 1],
            self.y_e.reshape((n, n, n), order="F")[:, :, n - 1],
            self.z_e.reshape((n, n, n), order="F")[:, :, n - 1],
            color="b",
            alpha=0.5,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show()

        return


# ===========================================================================


def apply_operators_3d(dr, ds, dt, x):

    "This function applies operators the same way as they are applied in NEK5000"
    "The only difference is that it is reversed, as this is python and we decided to leave that arrays as is"

    # Apply in r direction
    temp = x.reshape((int(x.size / dr.T.shape[0]), dr.T.shape[0])) @ dr.T

    # Apply in s direction
    temp = temp.reshape((ds.shape[1], ds.shape[1], int(temp.size / (ds.shape[1] ** 2))))
    ### The nek5000 way uses a for loop
    ## temp2 = np.zeros((ds.shape[1], ds.shape[0], int(temp.size/(ds.shape[1]**2)))) # This is needed because dimensions could reduce
    ## for k in range(0, temp.shape[0]):
    ##     temp2[k,:,:] = ds@temp[k,:,:]
    ### We can do it optimized in numpy if we reshape the operator. This way it can broadcast
    temp = ds.reshape((1, ds.shape[0], ds.shape[1])) @ temp

    # Apply in t direction
    temp = dt @ temp.reshape(dt.shape[1], (int(temp.size / dt.shape[1])))

    return temp.reshape(-1, 1)


def determine_initial_guess(self):

    if self.modal_search:
        # Find the closest gll point to the point of interest
        n = self.n
        distances = np.sqrt(
            (self.xj[0] - self.x_e.reshape((n, n, n), order="F")) ** 2
            + (self.yj[0] - self.y_e.reshape((n, n, n), order="F")) ** 2
            + (self.zj[0] - self.z_e.reshape((n, n, n), order="F")) ** 2
        )

        min_index_x = np.argmin(distances[:, 0, 0])
        min_index_y = np.argmin(distances[0, :, 0])
        min_index_z = np.argmin(distances[0, 0, :])

        # Knowing the indices, see what the rst values would be in a reference element
        self.rj[0] = self.x_gll[min_index_x]
        self.sj[0] = self.x_gll[min_index_y]
        self.tj[0] = self.x_gll[min_index_z]

        # print(self.rj[0],self.sj[0],self.tj[0])
    else:
        # If using lagrange basis, set initial guess to zero to avoid divide by zero
        self.rj[0] = 0
        self.sj[0] = 0
        self.tj[0] = 0

    return


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


def legendre_basis_at_xtest(n, xtest, use_scipy=False):

    m = len(xtest)  # Number of points

    # Allocate space
    Leg = np.zeros((n, m))

    if not use_scipy:

        # First row is filled with 1 according to recursive formula
        Leg[0, :] = np.ones((1, m))
        # Second row is filled with x according to recursive formula
        Leg[1, :] = np.multiply(np.ones((1, m)), xtest)

        # Apply the recursive formula for all x_i
        #### THE ROWS HERE ARE THE ORDERS!
        # look for recursive formula here if you want to verify https://en.wikipedia.org/wiki/Legendre_polynomials
        for j in range(1, n - 1):
            for k_ in range(0, m):
                Leg[j + 1, k_] = (
                    (2 * j + 1) * xtest[k_] * Leg[j, k_] - j * Leg[j - 1, k_]
                ) / (j + 1)

    elif use_scipy:

        for j in range(0, n):
            Leg[j, :] = np.polyval(legendre(j), xtest)[:]
            # Leg[j,:] = np.polyval(self.legendre[j], xtest)[:]

    return Leg


def legendre_basis_derivative_at_xtest(legtest, xtest, use_scipy=False):

    ## Now find the derivative matrix D_N,ij=(dpi_j/dxi)_at_xi=xi_i
    ##https://en.wikipedia.org/wiki/Legendre_polynomials

    n = legtest.shape[0]
    m = legtest.shape[1]

    # Allocate space
    D_N = np.zeros((n, m))

    if not use_scipy:

        # First row is filled with 1 according to recursive formula
        D_N[0, :] = np.zeros((1, m))
        # Second row is filled with x according to recursive formula
        D_N[1, :] = np.ones((1, m))

        for j in range(1, n - 1):
            for k_ in range(0, m):
                for p in range(j, 0 - 1, -2):
                    # if j==6 and k_==0: print(p)
                    D_N[j + 1, k_] += (
                        2 * legtest[p, k_] / (np.sqrt(2 / (2 * p + 1)) ** 2)
                    )

    elif use_scipy:

        for j in range(0, n):
            D_N[j, :] = np.polyval(legendre(j).deriv(), xtest)[:]
            # D_N[j,:] = np.polyval(self.legendre_prm[j], xtest)[:]

    return D_N


def orthonormalize(Leg):

    n = Leg.shape[0]
    m = Leg.shape[1]
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
    for i in range(0, m):
        for j in range(0, n):
            Leg[i, j] = Leg[i, j] * delta[j]
    return Leg.T


def get_reference_element(self):

    n = self.n

    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x_gll = np.flip(x)  # Quadrature
    w = np.flip(w_)  # Weights

    # Bounding boxes of the elements
    min_x = -1
    max_x = 1
    min_y = -1
    max_y = 1
    min_z = -1
    max_z = 1

    tmpx = np.zeros(len(x))
    tmpy = np.zeros(len(x))
    tmpz = np.zeros(len(x))
    for j in range(0, len(x)):
        tmpx[j] = (1 - x_gll[j]) / 2 * min_x + (1 + x_gll[j]) / 2 * max_x
        tmpy[j] = (1 - x_gll[j]) / 2 * min_y + (1 + x_gll[j]) / 2 * max_y
        tmpz[j] = (1 - x_gll[j]) / 2 * min_z + (1 + x_gll[j]) / 2 * max_z

    x_3d = np.kron(np.ones((n)), np.kron(np.ones((n)), tmpx))
    y_3d = np.kron(np.ones((n)), np.kron(tmpy, np.ones((n))))
    z_3d = np.kron(tmpz, np.kron(np.ones((n)), np.ones((n))))

    # Object atributes
    self.x_gll = x_gll
    self.w_gll = w
    self.x_e = x_3d.reshape(-1, 1)
    self.y_e = y_3d.reshape(-1, 1)
    self.z_e = z_3d.reshape(-1, 1)

    return


## Create transformation matrices for the element (Only needs to be done once)
def get_legendre_transformation_matrices(self):

    ## Legendre basis at the element gll points
    leg_gll = legendre_basis_at_xtest(self.n, self.x_gll)
    leg_prm_gll = legendre_basis_derivative_at_xtest(leg_gll, self.x_gll)

    ### Transformation matrices for the element (1D)
    v_1d = leg_gll.T
    v_1d_inv = np.linalg.inv(v_1d)
    d_1d = leg_prm_gll.T
    ### Transformation matrices in 2d
    v_2d = np.kron(v_1d, v_1d)
    v_2d_inv = np.kron(v_1d_inv, v_1d_inv)
    ### Transformation matrices in 3d
    v_3d = np.kron(v_1d, v_2d)
    v_3d_inv = np.kron(v_1d_inv, v_2d_inv)

    self.v1d = v_1d
    self.v1d_inv = v_1d_inv

    # Assign attributes
    self.v = v_3d
    self.v_inv = v_3d_inv

    return


# Standard Lagrange interpolation
def lagInterp_matrix_at_xtest(x, xTest):
    """
    Lagrange interpolation in 1D space
    """
    n = len(x)
    m = len(xTest)
    k = np.arange(n)
    Lk = np.zeros((n, m))
    for k_ in k:
        prod_ = 1.0
        for j in range(n):
            if j != k_:
                prod_ *= (xTest - x[j]) / (x[k_] - x[j])
        Lk[k_, :] = prod_

    return Lk


# Standard Lagrange interpolation
def lagInterp_derivative_matrix_at_xtest(x, Lk, xTest):
    """
    Lagrange interpolation in 1D space
    """
    n = len(x)
    k = np.arange(n)  # k and n are the same range
    m = len(xTest)
    # Lk=np.zeros((n,m))
    # for k_ in k:
    #    prod_=1.0
    #    for j in range(n):
    #        if j!=k_:
    #            prod_*=(xTest-x[j])/(x[k_]-x[j])
    #    Lk[k_,:]=prod_

    Lk_sum = np.zeros((n, m))
    for k_ in k:
        suma = 0.0
        for j in range(n):
            if j != k_:
                suma += 1 / (xTest - x[j])
        Lk_sum[k_, :] = suma

    Lk_prm = np.multiply(Lk_sum, Lk)

    return Lk_prm
