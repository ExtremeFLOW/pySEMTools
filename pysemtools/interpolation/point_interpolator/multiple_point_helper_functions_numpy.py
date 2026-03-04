"""Contains helper functions for multiple point interpolation using numpy"""

from math import pi
import numpy as np


def apply_operators_3d_einsum(dr, ds, dt, x):
    """

    This function applies operators the same way as they are applied in NEK5000

    The only difference is that it is reversed, as

    this is python and we decided to leave that arrays as is

    this function is more readable in sem.py, where tensor optimization is not used.

    """

    dshape = dr.shape
    xshape = x.shape
    xsize = xshape[2] * xshape[3]

    # Reshape the operator in the r direction
    drt = dr.transpose(
        0, 1, 3, 2
    )  # This is just the transpose of the operator, leaving the first dimensions unaltered
    drt_s0 = drt.shape[2]
    # drt_s1 = drt.shape[3]
    # Reshape the field to be consistent
    xreshape = x.reshape((xshape[0], xshape[1], int(xsize / drt_s0), drt_s0))
    # Apply the operator with einsum
    temp = np.einsum("ijkl,ijlm->ijkm", xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction
    temp = temp.reshape(
        (xshape[0], xshape[1], ds_s1, ds_s1, int(tempsize / (ds_s1**2)))
    )
    temp = np.einsum(
        "ijklm,ijkmn->ijkln", ds.reshape((dshape[0], dshape[1], 1, ds_s0, ds_s1)), temp
    )

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3] * temp.shape[4]
    # dt_s0 = dt.shape[2]
    dt_s1 = dt.shape[3]

    # Apply in t direction

    temp = temp.reshape((xshape[0], xshape[1], dt_s1, int(tempsize / dt_s1)))
    temp = np.einsum("ijkl,ijlm->ijkm", dt, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2] * temp.shape[3]

    return temp.reshape((tempshape[0], tempshape[1], tempsize, 1))

def apply_operators_3d(dr, ds, dt, x):
    """
    Applies operators in the r, s, and t directions (similar to NEK5000, but with a reversed order).

    Uses np.matmul for batched matrix multiplications in place of einsum.
    """
    # Ensure all arrays are contiguous to avoid unintended copies.
    dr = np.ascontiguousarray(dr)
    ds = np.ascontiguousarray(ds)
    dt = np.ascontiguousarray(dt)
    x = np.ascontiguousarray(x)

    # Save original shapes
    dshape = dr.shape
    xshape = x.shape
    # Total number of “points” in the last two dimensions
    xsize = xshape[2] * xshape[3]

    # --- Apply operator in the r direction ---
    # dr: shape (I, J, r0, r1) --> we need its transpose along the r matrix dims
    drt = dr.transpose(0, 1, 3, 2)  # shape becomes (I, J, r1, r0)
    drt_s0 = drt.shape[2]  # r1

    # Reshape the field so the matrix dimension aligns with drt
    # xreshape: (I, J, (xsize // r1), r1)
    xreshape = x.reshape(xshape[0], xshape[1], xsize // drt_s0, drt_s0)

    # Batched multiplication: for each (I, J) block,
    # multiply (xsize//r1 x r1) with (r1 x ?)
    temp = np.matmul(xreshape, drt)
    # Now temp has shape (I, J, (xsize // r1), ?)

    # --- Apply operator in the s direction ---
    # Get ds dimensions
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]
    # Combine the remaining dimensions of temp into a 5D array so that:
    #   axes 2 and 3 are reshaped to (ds_s1, ds_s1) and
    #   the last axis is the remainder (call it F)
    tempsize = temp.shape[2] * temp.shape[3]
    temp = temp.reshape(xshape[0], xshape[1], ds_s1, ds_s1, tempsize // (ds_s1**2))

    # ds needs to act on the “s” part.
    # Reshape ds to (I, J, 1, ds_s0, ds_s1) so that its last two dims form the matrix
    ds_reshaped = ds.reshape(dshape[0], dshape[1], 1, ds_s0, ds_s1)
    # Here, for each (I,J) and for each of the ds_s1 batch copies,
    # we multiply a matrix of shape (ds_s0, ds_s1) with one of shape (ds_s1, F)
    # np.matmul automatically broadcasts the singleton axis (1 → ds_s1)
    temp = np.matmul(ds_reshaped, temp)
    # The result has shape (I, J, ds_s1, ds_s0, F)

    # --- Apply operator in the t direction ---
    # Collapse the last three dimensions into one
    tempsize = temp.shape[2] * temp.shape[3] * temp.shape[4]
    # dt is assumed to have shape (I, J, t0, t1) with t1 used for contraction.
    dt_s1 = dt.shape[3]
    # Reshape temp so that its third dimension equals dt_s1 and the last dimension is the remainder
    temp = temp.reshape(xshape[0], xshape[1], dt_s1, tempsize // dt_s1)
    # Multiply: dt (shape: (I, J, t0, t1)) multiplies temp (shape: (I, J, t1, something))
    temp = np.matmul(dt, temp)
    # Now temp has shape (I, J, t0, something)

    # Flatten the last two dimensions and add a singleton fourth axis
    tempshape = temp.shape
    final_size = tempshape[2] * tempshape[3]
    return temp.reshape(tempshape[0], tempshape[1], final_size, 1)


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


def GLL_pwts(n, eps=10**-8, max_iter=1000):
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
    v = np.zeros((n, n))  # Legendre Vandermonde Matrix
    # Initial guess for the nodes: GLC points
    xi, _ = GLC_pwts(n)
    iter_ = 0
    err = 1000
    xi_old = xi
    while iter_ < max_iter and err > eps:
        iter_ += 1
        # Update the Legendre-Vandermonde matrix
        v[:, 0] = 1.0
        v[:, 1] = xi
        for j in range(2, n):
            v[:, j] = (
                (2.0 * j - 1) * xi * v[:, j - 1] - (j - 1) * v[:, j - 2]
            ) / float(j)
        # Newton-Raphson iteration
        xi = xi_old - (xi * v[:, n - 1] - v[:, n - 2]) / (n * v[:, n - 1])
        err = max(abs(xi - xi_old).flatten())
        xi_old = xi
    if iter_ > max_iter and err > eps:
        print("gllPts(): max iterations reached without convergence!")
    # Weights
    w = 2.0 / (n * (n - 1) * v[:, n - 1] ** 2.0)
    return xi, w


def legendre_basis_at_xtest_slow_obsolete(n, xtest):
    """
    The legendre basis depends on the element order and the points

    """

    m = xtest.shape[0]
    m2 = xtest.shape[1]

    # Allocate space
    leg = np.zeros((m, m2, n, 1))

    # First row is filled with 1 according to recursive formula
    leg[:, :, 0, 0] = np.ones((m, m2, 1, 1))[:, :, 0, 0]
    # Second row is filled with x according to recursive formula
    leg[:, :, 1, 0] = np.multiply(np.ones((m, m2, 1, 1)), xtest)[:, :, 0, 0]

    # Apply the recursive formula for all x_i
    # look for recursive formula here if you want to verify
    # https://en.wikipedia.org/wiki/Legendre_polynomials
    for j in range(1, n - 1):
        leg[:, :, j + 1, 0] = (
            (2 * j + 1) * xtest[:, :, 0, 0] * leg[:, :, j, 0] - j * leg[:, :, j - 1, 0]
        ) / (j + 1)

    return leg

def legendre_basis_at_xtest(n, xtest):
    """
    Compute the Legendre basis up to order n for a set of points.
    
    Parameters:
      n : int
          The number of Legendre polynomials to compute.
      xtest : np.ndarray
          Expected shape (m, m2, 1, 1) where the x-values are stored in xtest[:,:,0,0].
    
    Returns:
      leg : np.ndarray
          A tensor of shape (m, m2, n, 1) containing the Legendre polynomials.
    """
    m, m2 = xtest.shape[0], xtest.shape[1]
    # Extract the x values (assumed to be in the first element of the last two dims)
    x = xtest[:, :, 0, 0]
    
    # Preallocate array for Legendre polynomials
    leg = np.empty((m, m2, n), dtype=np.float64)
    
    # P_0(x) = 1
    leg[..., 0] = 1.0
    if n > 1:
        # P_1(x) = x
        leg[..., 1] = x
    
    # Recurrence: P_{j+1}(x) = ((2*j+1)*x*P_j(x) - j*P_{j-1}(x)) / (j+1)
    for j in range(1, n - 1):
        leg[..., j+1] = ((2 * j + 1) * x * leg[..., j] - j * leg[..., j-1]) / (j + 1)
    
    # Add back the singleton dimension to match the expected output shape
    return leg[..., np.newaxis]

def legendre_basis_derivative_at_xtest_slow_obsolete(legtest, xtest):
    """

    This is a slow implementaiton with a slow recursion. It does not need
    special treatmet at the boundary

    """

    ## Now find the derivative matrix D_N,ij=(dpi_j/dxi)_at_xi=xi_i
    ##https://en.wikipedia.org/wiki/Legendre_polynomials

    m = legtest.shape[0]
    m2 = xtest.shape[1]
    n = legtest.shape[2]

    # Allocate space
    d_n = np.zeros((m, m2, n, 1))

    # For first polynomial: derivatice is 0
    d_n[:, :, 0, 0] = np.zeros((m, m2, 1, 1))[:, :, 0, 0]
    # For second polynomial: derivatice is 1
    d_n[:, :, 1, 0] = np.ones((m, m2, 1, 1))[:, :, 0, 0]

    for j in range(1, n - 1):
        for p in range(j, 0 - 1, -2):
            d_n[:, :, j + 1, 0] += (
                2 * legtest[:, :, p, 0] / (np.sqrt(2 / (2 * p + 1)) ** 2)
            )

    return d_n

def legendre_basis_derivative_at_xtest(legtest, xtest):
    """
    Compute the derivative matrix D_N, where D_N[..., j] = (dP_j/dx)_at_x,
    using a vectorized weighted sum over the Legendre basis values.
    
    Parameters:
      legtest : np.ndarray
          A tensor of shape (m, m2, n, 1) containing the Legendre polynomials.
      xtest : np.ndarray
          A tensor whose second dimension gives m2 (used only to get m2 here).
    
    Returns:
      d_n : np.ndarray
          A tensor of shape (m, m2, n, 1) containing the derivatives.
    """
    m, m2, n, _ = legtest.shape
    
    # Preallocate derivative array.
    d_n = np.zeros((m, m2, n), dtype=np.float64)
    
    # Derivative of P_0 is 0 (already set) and of P_1 is 1.
    if n > 1:
        d_n[..., 1] = 1.0

    if n <= 2:
        return d_n[..., np.newaxis]
    
    # Build a weight matrix M for j = 1, ..., n-2.
    # For a given j (which will fill d_n[..., j+1]),
    # valid p indices are those satisfying: 0 <= p <= j and (j-p) even.
    n_out = n - 2  # corresponds to orders 2 through n-1.
    j_indices = np.arange(1, n - 1).reshape(n_out, 1)  # shape: (n_out, 1)
    p_indices = np.arange(n).reshape(1, n)             # shape: (1, n)
    valid_mask = (p_indices <= j_indices) & (((j_indices - p_indices) % 2) == 0)
    # Compute weights (2p+1) for each valid p.
    weights = (2 * p_indices + 1).astype(np.float64)
    M = weights * valid_mask.astype(np.float64)  # shape: (n_out, n)
    
    # Remove the trailing singleton dimension from legtest.
    legtest_squeezed = legtest[..., 0]  # shape: (m, m2, n)
    # Flatten the first two dimensions to combine m and m2.
    legtest_flat = legtest_squeezed.reshape(-1, n).T  # shape: (n, m*m2)
    
    # Compute the weighted sum: for each output order, sum_{p} M[j, p] * legtest[..., p]
    # This yields an array of shape (n_out, m*m2)
    d_result = M @ legtest_flat  
    # Reshape back to (m, m2, n_out)
    d_result = d_result.T.reshape(m, m2, n_out)
    
    # Place the computed derivative results into orders 2 to n-1.
    d_n[..., 2:] = d_result
    return d_n[..., np.newaxis]

def lag_interp_matrix_at_xtest(x, xtest):
    """

    Lagrange interpolation in 1D space

    """

    n = x.shape[0]
    m = xtest.shape[0]
    m2 = xtest.shape[1]
    # k=np.arange(n)

    # Allocate space
    lk = np.zeros((m, m2, n, 1))

    for k_ in range(0, n):
        prod_ = np.ones((m, m2, 1, 1))
        for j in range(n):
            if j != k_:
                prod_ = (
                    prod_
                    * (xtest[:, :, :, :] - x[j, :, :, :])
                    / (x[k_, :, :, :] - x[j, :, :, :])
                )
        lk[:, :, k_, :] = prod_[:, :, 0, :]

    return lk

def bar_interp_matrix_at_xtest(x, xtest, w = None):
    """
    Compute the Lagrange interpolation matrix using the barycentric interpolation formula. 
    """
    # Remove singleton dimensions so that x becomes a 1D array of nodes.
    x_vec = x[:, 0, 0, 0]
    xtest_mat = xtest[..., 0, 0]
    n = x_vec.shape[0]
    
    # Compute the barycentric weights if they are not provided only
    if w is None:
        # Compute the barycentric weights - maybe have them as an input
        # For each k, compute: w[k] = 1 / ∏_{j ≠ k} (x_vec[k] - x_vec[j])
        diff = x_vec[:, None] - x_vec[None, :]  # shape: (n, n)
        np.fill_diagonal(diff, 1.0)
        w = 1.0 / np.prod(diff, axis=1)  # shape: (n,)
    
    # Compute differences between each test point and every node.
    diff_xtest = xtest_mat[..., None] - x_vec[None, None, :]

    # If the point to interpolate coincides with one of the nodes, it will 
    # proruce an error. Use this to supress it 
    with np.errstate(divide='ignore', invalid='ignore'):
 
        # Apply the barycentric formula
        num = w / diff_xtest 
        denom = np.sum(num, axis=-1, keepdims=True)
        L = num / denom

    # This is risky. If you see interpolation errors in the future,
    # the use the actual check in the else bloc. Otherwise use the full
    # lagrange matrix construction in the typical way
    if 1 == 1:
        L[np.isnan(L)] = 1.0
    else:
        # Handle the points that coincide
        exact_match = np.isclose(diff_xtest, 0)  # shape: (npts, nelems, n)
        has_exact = np.any(exact_match, axis=-1)   # shape: (npts, nelems)
        if np.any(has_exact):
            idx_exact = np.argmax(exact_match, axis=-1)
            # Identify the points that match
            I, J = np.nonzero(has_exact)
            # Only for those points set the basis to 0 except at the matching node
            L[I, J, :] = 0
            L[I, J, idx_exact[I, J]] = 1.0

    # Expand the last dimension for consistency
    return L[..., None]

def get_reference_element(self):
    """
    This routine creates a reference element containing all GLL points.

    It also creates a place holder where data from future elements can be stored.

    Data is always of shape [points, elements, rows, columns]

    """

    n = self.n
    max_pts = self.max_pts
    max_elems = self.max_elems

    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x_gll = np.copy(np.flip(x))  # Quadrature
    w = np.copy(np.flip(w_))  # Weights

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
    ## Allocate
    self.x_gll = np.zeros((n, 1, 1, 1))
    self.w_gll = np.zeros((n, 1, 1, 1))
    self.x_e = np.zeros((max_pts, max_elems, x_3d.size, 1))
    self.y_e = np.zeros((max_pts, max_elems, y_3d.size, 1))
    self.z_e = np.zeros((max_pts, max_elems, z_3d.size, 1))
    ## Assing values
    self.x_gll[:, 0, 0, 0] = x_gll
    self.w_gll[:, 0, 0, 0] = w
    self.x_e[:, :] = x_3d.reshape(-1, 1)
    self.y_e[:, :] = y_3d.reshape(-1, 1)
    self.z_e[:, :] = z_3d.reshape(-1, 1)

    return


## Create transformation matrices for the element (Only needs to be done once)
def get_basis_transformation_matrices(self):
    """
    This routines generates the transformation matrices to be applied directly to the data.

    In principle this is not needed, but since many transformations need to be made, we store it.

    """

    ## Legendre basis at the element gll points
    leg_gll = legendre_basis_at_xtest(self.n, self.x_gll)
    leg_prm_gll = legendre_basis_derivative_at_xtest(leg_gll, self.x_gll)

    ## For the sake of simplicity, reshape the arrays only for this routine
    leg_gll = (leg_gll.transpose(3, 1, 2, 0)).reshape((self.n, self.n))
    leg_prm_gll = (leg_prm_gll.transpose(3, 1, 2, 0)).reshape((self.n, self.n))

    ## Transformation matrices for the element (1D)
    v_1d = leg_gll.T
    v_1d_inv = np.linalg.inv(v_1d)
    # d_1d = leg_prm_gll.T
    ## Transformation matrices in 2d
    v_2d = np.kron(v_1d, v_1d)
    v_2d_inv = np.kron(v_1d_inv, v_1d_inv)
    ## Transformation matrices in 3d
    v_3d = np.kron(v_1d, v_2d)
    v_3d_inv = np.kron(v_1d_inv, v_2d_inv)

    self.v1d = v_1d.reshape((1, 1, self.n, self.n))
    self.v1d_inv = v_1d_inv.reshape((1, 1, self.n, self.n))

    # Assign attributes
    self.v = v_3d.reshape((1, 1, self.n**3, self.n**3))
    self.v_inv = v_3d_inv.reshape((1, 1, self.n**3, self.n**3))

    return
