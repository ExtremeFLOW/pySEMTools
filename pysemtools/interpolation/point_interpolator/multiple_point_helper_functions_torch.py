""" helper functions for multiple point interpolation using pytorch"""

import numpy as np
import torch

# from .multiple_point_helper_functions_numpy import GLC_pwts, GLL_pwts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def apply_operators_3d_einsum(dr, ds, dt, x):
    """

    This function applies operators the same way as they are applied in NEK5000

    The only difference is that it is reversed, as this is
    python and we decided to leave that arrays as is

    this function is more readable in sem.py, where tensor optimization is not used.

    """

    dshape = dr.shape
    xshape = x.shape
    xsize = xshape[2] * xshape[3]

    # Reshape the operator in the r direction
    drt = dr.permute(
        0, 1, 3, 2
    )  # This is just the transpose of the operator, leaving the first dimensions unaltered
    drt_s0 = drt.shape[2]
    # drt_s1 = drt.shape[3]
    # Reshape the field to be consistent
    xreshape = x.reshape((xshape[0], xshape[1], int(xsize / drt_s0), drt_s0))
    # Apply the operator with einsum
    temp = torch.einsum("ijkl,ijlm->ijkm", xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction
    temp = temp.reshape(
        (xshape[0], xshape[1], ds_s1, ds_s1, int(tempsize / (ds_s1**2)))
    )
    temp = torch.einsum(
        "ijklm,ijkmn->ijkln", ds.reshape((dshape[0], dshape[1], 1, ds_s0, ds_s1)), temp
    )

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3] * temp.shape[4]
    # dt_s0 = dt.shape[2]
    dt_s1 = dt.shape[3]

    # Apply in t direction

    temp = temp.reshape((xshape[0], xshape[1], dt_s1, int(tempsize / dt_s1)))
    temp = torch.einsum("ijkl,ijlm->ijkm", dt, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2] * temp.shape[3]

    return temp.reshape((tempshape[0], tempshape[1], tempsize, 1))

def apply_operators_3d(dr, ds, dt, x):
    """
    Applies operators in the r, s, and t directions (similar to NEK5000, but with a reversed order)
    using torch.matmul instead of torch.einsum.
    """
    dshape = dr.shape  # expected shape: (I, J, dr_dim0, dr_dim1)
    xshape = x.shape   # expected shape: (I, J, X, Y)
    xsize = xshape[2] * xshape[3]

    # --- r Direction ---
    # Transpose dr so that the last two dims swap, shape becomes (I, J, dr_dim1, dr_dim0)
    drt = dr.permute(0, 1, 3, 2)
    drt_s0 = drt.shape[2]
    # Reshape x so its last dimension matches drt's contracting dimension
    # xreshape will have shape: (I, J, (xsize // drt_s0), drt_s0)
    xreshape = x.reshape(xshape[0], xshape[1], int(xsize / drt_s0), drt_s0)
    # Batched matrix multiply: (I, J, M, drt_s0) @ (I, J, drt_s0, drt_dim?) -> (I, J, M, drt_dim?)
    temp = torch.matmul(xreshape, drt)

    # --- s Direction ---
    # Get dimensions from ds
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]
    # Current temp is (I, J, M, drt_dim?); combine the last two dims:
    tempsize = temp.shape[2] * temp.shape[3]
    # Reshape temp so that the two dimensions become (ds_s1, ds_s1, F)
    temp = temp.reshape(xshape[0], xshape[1], ds_s1, ds_s1, int(tempsize / (ds_s1**2)))
    # Reshape ds for broadcasting: from (I, J, ds_s0, ds_s1) to (I, J, 1, ds_s0, ds_s1)
    ds_reshaped = ds.reshape(dshape[0], dshape[1], 1, ds_s0, ds_s1)
    # Batched matrix multiplication:
    #   ds_reshaped: (I, J, 1, ds_s0, ds_s1)
    #   temp:        (I, J, ds_s1, ds_s1, F)
    # matmul will contract over the last dim of ds_reshaped and the third dim of temp,
    # giving a result of shape (I, J, ds_s1, ds_s0, F)
    temp = torch.matmul(ds_reshaped, temp)

    # --- t Direction ---
    # Collapse the last three dimensions into two:
    tempsize = temp.shape[2] * temp.shape[3] * temp.shape[4]
    dt_s1 = dt.shape[3]
    # Reshape temp to have shape (I, J, dt_s1, remaining)
    temp = temp.reshape(xshape[0], xshape[1], dt_s1, int(tempsize / dt_s1))
    # dt has shape (I, J, dt_dim0, dt_s1); perform batched matrix multiply:
    #   (I, J, dt_dim0, dt_s1) @ (I, J, dt_s1, something) -> (I, J, dt_dim0, something)
    temp = torch.matmul(dt, temp)

    # Flatten the last two dimensions and add a singleton final dimension
    tempshape = temp.shape
    final_size = temp.shape[2] * temp.shape[3]
    return temp.reshape(tempshape[0], tempshape[1], final_size, 1)

def legendre_basis_at_xtest_slow_obsolete(n, xtest):
    """
    The legendre basis depends on the element order and the points

    """
    m = xtest.shape[0]
    m2 = xtest.shape[1]

    # Allocate space
    leg = torch.zeros((m, m2, n, 1), dtype=torch.float64, device=device)

    # First row is filled with 1 according to recursive formula
    leg[:, :, 0, 0] = 1.0

    # Second row is filled with x according to recursive formula
    leg[:, :, 1, 0] = xtest[:, :, 0, 0]

    # Apply the recursive formula for all x_i
    # look for recursive formula here if you want to verify
    # https://en.wikipedia.org/wiki/Legendre_polynomials

    ## Here we perform the extra step of cloning the tensor to not have trouble
    ## with inplace modifications in case we want to use autograd for xtest.
    for j in range(1, n - 1):
        leg_j_p1 = (
            (2 * j + 1) * xtest[:, :, 0, 0] * leg[:, :, j, 0] - j * leg[:, :, j - 1, 0]
        ) / (j + 1)
        leg = leg.clone()
        leg[:, :, j + 1, 0] = leg_j_p1

    return leg

def legendre_basis_at_xtest(n, xtest):
    """
    Compute Legendre basis functions up to order n for the given x values in a torch-friendly manner.
    This version avoids in-place modifications by accumulating each polynomial in a list.
    
    Parameters:
      n : int
          The number of Legendre polynomials to compute.
      xtest : torch.Tensor
          Expected shape (m, m2, 1, 1), with the x-values stored in xtest[:, :, 0, 0].
    
    Returns:
      leg : torch.Tensor
          A tensor of shape (m, m2, n, 1) with the computed Legendre polynomials.
    """
    m, m2 = xtest.shape[0], xtest.shape[1]
    # Extract x (assumed to be in the first element of the last two dims)
    x = xtest[:, :, 0, 0]
    
    # Use a list to accumulate the polynomials (functional style avoids in-place modifications)
    polys = []
    # P_0(x) = 1
    polys.append(torch.ones_like(x))
    if n > 1:
        # P_1(x) = x
        polys.append(x)
    
    # Recurrence: P_{j+1}(x) = ((2*j+1)*x*P_j(x) - j*P_{j-1}(x)) / (j+1)
    for j in range(1, n - 1):
        p_next = ((2 * j + 1) * x * polys[j] - j * polys[j - 1]) / (j + 1)
        polys.append(p_next)
    
    # Stack along a new dimension to form a tensor of shape (m, m2, n)
    leg = torch.stack(polys, dim=2)
    # Restore the singleton trailing dimension: shape (m, m2, n, 1)
    return leg.unsqueeze(-1)

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
    d_n = torch.zeros((m, m2, n, 1), dtype=torch.float64, device=device)

    # For first polynomial: derivatice is 0
    d_n[:, :, 0, 0] = torch.zeros((m, m2, 1, 1), dtype=torch.float64, device=device)[
        :, :, 0, 0
    ]
    # For second polynomial: derivatice is 1
    d_n[:, :, 1, 0] = torch.ones((m, m2, 1, 1), dtype=torch.float64, device=device)[
        :, :, 0, 0
    ]

    for j in range(1, n - 1):
        for p in range(j, 0 - 1, -2):
            d_n[:, :, j + 1, 0] += (
                2 * legtest[:, :, p, 0] / (torch.sqrt(torch.tensor(2 / (2 * p + 1))) ** 2)
            )

    return d_n

def legendre_basis_derivative_at_xtest(legtest, xtest):
    """
    Compute the derivative matrix D_N where D_N[i,j] = (dP_j/dx)_at_xi,
    using a vectorized approach that avoids Python loops.

    Parameters:
      legtest: torch.Tensor of shape (m, m2, n, 1)
      xtest:   torch.Tensor with shape whose second dimension gives m2

    Returns:
      d_n: torch.Tensor of shape (m, m2, n, 1)
    """
    # Get dimensions
    m = legtest.shape[0]
    m2 = xtest.shape[1]
    n = legtest.shape[2]

    # Allocate output tensor (initialize to zero)
    d_n = torch.zeros((m, m2, n, 1), dtype=torch.float64, device=legtest.device)

    # For the first polynomial, derivative is 0 (already zero).
    # For the second polynomial, derivative is 1.
    d_n[:, :, 1, 0] = 1.0

    # If n <= 2, there is nothing more to do.
    if n <= 2:
        return d_n

    # Precompute the weight matrix for j=1,...,n-2 (which fill d_n[:,:,j+1,0]).
    # For each outer loop index j in the original code (j in 1,..., n-2),
    # valid p are those with 0 <= p <= j and (j - p) even.
    # We'll create a matrix M of shape (n-2, n) such that:
    #    M[j-1, p] = (2*p+1) if (p <= j and (j-p) % 2 == 0), else 0.
    n_out = n - 2  # number of outputs computed by the loop
    # j_indices will correspond to original j = 1,..., n-2.
    j_indices = torch.arange(1, n - 1, device=legtest.device, dtype=torch.int64).unsqueeze(1)  # shape (n_out, 1)
    p_indices = torch.arange(n, device=legtest.device, dtype=torch.int64).unsqueeze(0)           # shape (1, n)
    # Create mask: valid if p <= j and (j - p) is even.
    valid_mask = (p_indices <= j_indices) & (((j_indices - p_indices) % 2) == 0)
    # Compute weights: (2*p + 1) for each p.
    weights = (2 * p_indices + 1).to(torch.float64)  # shape (1, n)
    # Weight matrix M: shape (n_out, n)
    M = weights * valid_mask.to(torch.float64)

    # Now, legtest has shape (m, m2, n, 1); squeeze the last dimension:
    legtest_squeezed = legtest.squeeze(-1)  # shape (m, m2, n)
    # We need to apply the weighted sum to each (m, m2) pair.
    # Reshape legtest to (m*m2, n) for batched matmul.
    legtest_flat = legtest_squeezed.reshape(m * m2, n).transpose(0, 1)  # shape (n, m*m2)
    # Compute the weighted sum for each output index:
    #   d_result will have shape (n_out, m*m2)
    d_result = torch.matmul(M, legtest_flat)
    # Reshape d_result back to (m, m2, n_out)
    d_result = d_result.transpose(0, 1).reshape(m, m2, n_out)
    # Place the computed results into d_n for indices 2 to n-1.
    d_n[:, :, 2:n, 0] = d_result

    return d_n

def lag_interp_matrix_at_xtest(x, xtest):
    """

    Lagrange interpolation in 1D space

    """

    n = x.shape[0]
    m = xtest.shape[0]
    m2 = xtest.shape[1]
    # k=np.arange(n)

    # Allocate space
    lk = torch.zeros((m, m2, n, 1), dtype=torch.float64, device=device)

    for k_ in range(0, n):
        prod_ = torch.ones((m, m2, 1, 1), dtype=torch.float64, device=device)
        for j in range(n):
            if j != k_:
                prod_ = (
                    prod_
                    * (xtest[:, :, :, :] - x[j, :, :, :])
                    / (x[k_, :, :, :] - x[j, :, :, :])
                )
        lk[:, :, k_, :] = prod_[:, :, 0, :]

    return lk
