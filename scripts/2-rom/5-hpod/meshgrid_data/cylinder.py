import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

def trap_weights_1d(x):
    """Trapezoidal-rule weights for possibly nonuniform 1D grid x (size n)."""
    x = np.asarray(x)
    n = x.size
    w = np.zeros(n, dtype=float)
    if n == 1:
        w[0] = 0.0
        return w
    dx = np.diff(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if n > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w

def smoothstep(t):
    """C^1 smoothstep from 0 to 1 on t∈[0,1]."""
    t = np.clip(t, 0.0, 1.0)
    return 3*t*t - 2.0*t*t*t

def plot_structured_grid(ax, X2, Y2, stride_i=3, stride_j=3, title=""):
    nx, ny = X2.shape
    for i in range(0, nx, stride_i):
        ax.plot(X2[i, :], Y2[i, :], linewidth=0.6)
    for j in range(0, ny, stride_j):
        ax.plot(X2[:, j], Y2[:, j], linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def mass_matrix(Yd, x_1d, y_1d, z_1d, yc=0.0):
    """
    Diagonal mass weights w[i,j,k]

    Assumes mapping:
      X = x (unchanged), Z = z (unchanged), Y = Yd(x,y,z)
    so J = dY/dy (w.r.t computational y coordinate y_1d).

    """
    
    # 1D trap weights (works for uniform/nonuniform)
    wx = trap_weights_1d(x_1d)   
    wy = trap_weights_1d(y_1d)   
    wz = trap_weights_1d(z_1d)   
    if np.sum(wz) <= 1e-8: # Then I am using only a 2D slice
        wz = np.ones_like(wz)

    # Base tensor-product weights
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]

    # Split indices
    lower = np.where(y_1d < yc)[0]
    upper = np.where(y_1d > yc)[0]
    equal = np.where(np.isclose(y_1d, yc))[0]

    # Lower half Jacobian: gradient only on that subarray
    if lower.size >= 2:
        J_lo = np.gradient(Yd[:, lower, :], y_1d[lower], axis=1)
        w[:, lower, :] *= J_lo
    elif lower.size == 1:
        # too few points to differentiate; safest: zero it out
        w[:, lower, :] *= 0.0

    # Upper half Jacobian
    if upper.size >= 2:
        J_up = np.gradient(Yd[:, upper, :], y_1d[upper], axis=1)
        w[:, upper, :] *= J_up
    elif upper.size == 1:
        w[:, upper, :] *= 0.0

    # If you have a gridline exactly at yc, set its weight to 0 (it lies on the cut)
    if equal.size:
        w[:, equal, :] = 0.0

    return w

def create_cylinder(
        X, Y,
        x_bbox, y_bbox,
        center=(0,0),
        x_smoothing_npoints = 1,
        R=1,
        y_smooth_window = [1,5],
):

    # =======================================================
    # Obtain the required displacement to create the cylinder
    # =======================================================

    # Perform some checks    
    xc, yc = center
    y_min, y_max = y_bbox

    # computational dy (minimum)
    if Y.ndim == 2:
        # Y is built from y_1d; use min dy from the first x-line
        dy_comp = float(np.min(np.diff(Y[0, :])))
    else:
        dy_comp = float(np.min(np.diff(Y[0, :, 0])))

    if dy_comp <= 0:
        raise ValueError("Computational y grid must be strictly increasing.")

    dx = X - xc # Translated coordinates to origin at center
    disps = np.sqrt(np.clip(R*R - dx*dx, 0.0, None)) # This is the absolute value of y that every point should be displaced to create the cylinder

    # Mask what is in the top and bottom of the domain
    mask_up = (Y >= yc)
    mask_lo = ~mask_up

    # ===============================================================================
    # Get a smooth step function in the y direction to avoid deforming the boundaries
    # ===============================================================================

    y_1d = Y[0,:]
    s_1d = np.zeros_like(y_1d)
    mask_up_1d = (y_1d >= yc)
    mask_lo_1d = ~mask_up_1d
    y_up_1d = y_1d[mask_up_1d]
    y_lo_1d = y_1d[mask_lo_1d]

    # Set the new coordinate system from 1 - 0
    smooth_window = y_smooth_window
    w0, w1 = smooth_window
    r = w1 - w0 # Range
    s_up = np.ones_like(y_up_1d)
    mid = (y_up_1d > w0) & (y_up_1d < w1) # Mask that marks where smoothing happens
    s_up[y_up_1d >= w1] = 0.0
    s_up[mid] = 1.0 - (y_up_1d[mid] - w0)/r  # map y to [0,1] inside the smoothing window and substract to ramp down
    s_up = np.clip(s_up, 0.0, 1.0) # 1 close to cylinder, 0 far away, and linear in-between
    
    # Mirror the values on the lower half of the domain
    s_1d[mask_up_1d] = s_up
    s_1d[mask_lo_1d] = np.copy(np.flip(s_up))
    # Smooth step function
    smooth_y = smoothstep(s_1d)
    
    # Apply the smoothing
    disps = disps * smooth_y[None, :]

    # ========================================================
    # Smooth in x too, with as imple convolution sort of thing
    # ========================================================

    if x_smoothing_npoints > 1:
        kernel_lenght = x_smoothing_npoints
        kernel = np.ones(kernel_lenght)
        disp_conv = np.zeros_like(disps)
        for k in range(X.shape[2]):
            for j in range(Y.shape[1]):
                disp_conv[:, j, k] = np.convolve(disps[:, j, k], kernel, mode='same') / kernel_lenght

        for k in range(X.shape[2]):
            for j in range(Y.shape[1]):
                for i in range(X.shape[0]): 
                    disps[i, j, k] = max(disp_conv[i, j, k], disps[i, j, k])
                    #disps[i, j, k] = disp_conv[i, j, k]

    # =======================
    # Apply the displacements
    # =======================
    # Perform the displacements
    Yd = Y.copy()
    Yd[mask_up] = Yd[mask_up] + disps[mask_up]
    Yd[mask_lo] = Yd[mask_lo] - disps[mask_lo]


    return X, Yd

# ===========================

# Domain
x_bbox = [-10, 30]
y_bbox = [-15, 15]
z_bbox = [2.5 , 2.5]
# Resolution
ddx = 0.1
ddy = 0.1
ddz = 1.0
# Cylinder info
center = (0.0, 0.0)
R = 1.0
x_kernel_support = 1 # This indicates the support in physical units of the convolution kernel used to smooth in the x direction
x_smoothing_npoints = int(x_kernel_support/ddx)
y_smooth_window = [1,5] # This indicates where the smooth step function is valid in the y direction



if __name__ == "__main__":

    # Number of points (from resolution and domain)
    nx = int((x_bbox[1] - x_bbox[0]) / ddx)
    ny = int((y_bbox[1] - y_bbox[0]) / ddy)
    if np.mod(nx, 2) == 0:
        nx += 1  # make it odd
    if np.mod(ny, 2) != 0:
        ny += 1  # make it even
    nz = 1
    ## 1D grids
    x_1d = np.linspace(*x_bbox, nx)
    y_1d = np.linspace(*y_bbox, ny)
    z_1d = np.linspace(*z_bbox, nz)

    # 3D cube
    X, Y, Z = np.meshgrid(x_1d, y_1d, z_1d, indexing="ij")  # (nx, ny, nz)

    # Cylinder center (all through the z direction)
    Xd, Yd = create_cylinder(X, Y, x_bbox, y_bbox, center=center, R=R, x_smoothing_npoints=x_smoothing_npoints, y_smooth_window=y_smooth_window)
    Zd = Z

    if 1 == 1:
        # ---------------------------
        # Plot one z-slice (e.g. mid-plane)
        # ---------------------------
        k = nz // 2  # slice index
        theta = np.linspace(0, 2*np.pi, 400)
        cx = center[0] + R*np.cos(theta)
        cy = center[1] + R*np.sin(theta)

        fig2, ax2 = plt.subplots()
        plot_structured_grid(ax2, Xd[:, :, k], Yd[:, :, k], stride_i=1, stride_j=1, title="Deformed grid (z-slice)")
        ax2.plot(cx, cy, linewidth=1.2)
        plt.show()

        # ---------------------------
        # Sanity checks
        # ---------------------------
        rmin = np.min(np.sqrt((Xd-center[0])**2 + (Yd-center[1])**2))
        print("min radius after deformation =", rmin)
        print("X unchanged:", np.allclose(Xd, X))
        print("Shapes:", X.shape, Y.shape, Z.shape)

    # Generate mass matrix
    B =  mass_matrix(Yd, x_1d, y_1d, z_1d, yc=center[1])
    B = np.ascontiguousarray(B)
    print("Sum of mass matrix:", np.sum(B))

    cube = (x_bbox[1] - x_bbox[0]) * (y_bbox[1] - y_bbox[0]) * (z_bbox[1] - z_bbox[0])

    if (z_bbox[1] - z_bbox[0]) > 1e-8:
        vol_exact = cube - np.pi * R * R * (z_bbox[1] - z_bbox[0])
        print("Exact volume =", vol_exact)
    else:
        area_exact = (x_bbox[1] - x_bbox[0]) * (y_bbox[1] - y_bbox[0]) - np.pi * R * R
        print("Exact area =", area_exact)

        if nz > 1:
            raise ValueError("z_1d is just one value. If you want a 2d slice, then make sure to make nz = 1")

    print(Xd.shape, Yd.shape, Zd.shape, B.shape)

    fname = 'points.hdf5'
    with h5py.File(fname, 'w') as f:

        # Create a header
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['nz'] = nz
        #f.attrs['probe_list_key'] = 'xyz'

        # Include data sets
        f.create_dataset('x', data=Xd)
        f.create_dataset('y', data=Yd)
        f.create_dataset('z', data=Zd)
        f.create_dataset('mass', data=B)
