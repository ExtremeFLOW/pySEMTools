""" Module that contains classes and methods that provide standard usefull quantities in SEM"""

from math import pi
import numpy as np
import sys
if 'torch' in sys.modules:
    import torch
else:
    torch = None
from ..monitoring.logger import Logger

__all__ = ['Coef']


class Coef:
    """
    Class that contains arrays like mass matrix, jacobian, jacobian inverse, etc.

    This class can be used when mathematical operations such as derivation and integration is needed on the sem mesh.

    Parameters
    ----------
    msh : Mesh
        Mesh object.

    comm : Comm
        MPI comminicator object.

    get_area : bool, optional
        If True, the area integration weight and normal vectors will be calculated. (Default value = True).

    apply_1d_operators : bool, optional
        If True, the 1D operators will be applied instead of building 3D operators. (Default value = True).

    Attributes
    ----------
    drdx : ndarray
        component [0,0] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    drdy : ndarray
        component [0,1] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    drdz : ndarray
        component [0,2] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dsdx : ndarray
        component [1,0] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dsdy : ndarray
        component [1,1] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dsdz : ndarray
        component [1,2] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dtdx : ndarray
        component [2,0] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dtdy : ndarray
        component [2,1] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    dtdz : ndarray
        component [2,2] of the jacobian inverse tensor for each point. shape is (nelv, lz, ly, lx).
    B : ndarray
        Mass matrix for each point. shape is (nelv, lz, ly, lx).
    area : ndarray
        Area integration weight for each point in the facets. shape is (nelv, 6, ly, lx).
    nx : ndarray
        x component of the normal vector for each point in the facets. shape is (nelv, 6, ly, lx).
    ny : ndarray
        y component of the normal vector for each point in the facets. shape is (nelv, 6, ly, lx).
    nz : ndarray
        z component of the normal vector for each point in the facets. shape is (nelv, 6, ly, lx).

    Returns
    -------

    Examples
    --------
    Assuming you have a mesh object and MPI communicator object, you can initialize the Coef object as follows:

    >>> from pysemtools import Coef
    >>> coef = Coef(msh, comm)
    """

    def __init__(self, msh, comm, get_area=False, apply_1d_operators=True, bckend = "numpy"):

        self.log = Logger(comm=comm, module_name="Coef")
        self.log.tic()

        self.log.write("info", "Initializing Coef object")
        self.log.write("info", "Getting derivative matrices")

        self.gdim = msh.gdim
        self.dtype = msh.x.dtype
        self.apply_1d_operators = apply_1d_operators

        self.bckend = bckend
        if bckend == 'torch':

            if sys.modules.get("torch") is None:
                raise ImportError("torch is not installed/imported. Please install it and import it in your driver python script to use the torch backend.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Correct dtype if input is torch tensor
            if self.dtype == torch.float64:
                self.dtype = np.float64
            elif self.dtype == torch.float32:
                self.dtype = np.float32
            # Set the actual torch dtype in another place
            if self.dtype == np.float64:
                self.dtype_d = torch.float64
            elif self.dtype == np.float32:
                self.dtype_d = torch.float32
            if not self.apply_1d_operators:
                raise ValueError("The torch backend only supports the apply_1d_operators option.")

        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            msh.lx, msh.gdim, apply_1d_operators=apply_1d_operators, dtype=self.dtype
        )

        self.dr, self.ds, self.dt, self.dn = get_derivative_matrix(
            msh.lx, msh.gdim, dtype=self.dtype, apply_1d_operators=apply_1d_operators
        )

        # Take the data to the GPU
        if bckend == 'torch':
            self.v = torch.as_tensor(self.v, dtype=self.dtype_d, device=self.device)
            self.vinv = torch.as_tensor(self.vinv, dtype=self.dtype_d, device=self.device)
            self.w3 = torch.as_tensor(self.w3, dtype=self.dtype_d, device=self.device)
            self.x = torch.as_tensor(self.x.copy(), dtype=self.dtype_d, device=self.device)
            self.w = torch.as_tensor(self.w.copy(), dtype=self.dtype_d, device=self.device)
            self.dr = torch.as_tensor(self.dr, dtype=self.dtype_d, device=self.device)
            self.ds = torch.as_tensor(self.ds, dtype=self.dtype_d, device=self.device)
            if msh.gdim > 2:
                self.dt = torch.as_tensor(self.dt, dtype=self.dtype_d, device=self.device)
            self.dn = torch.as_tensor(self.dn, dtype=self.dtype_d, device=self.device)

        self.log.write("info", "Calculating the components of the jacobian")

        # Find the components of the jacobian per point
        # jac(x,y,z) = [dxdr, dxds, dxdt ; dydr, dyds, dydt; dzdr, dzds, dzdt]
        self.dxdr = self.dudrst(msh.x, direction="r")
        self.dxds = self.dudrst(msh.x, direction="s")
        if msh.gdim > 2:
            self.dxdt = self.dudrst(msh.x, direction="t")

        self.dydr = self.dudrst(msh.y, direction="r")
        self.dyds = self.dudrst(msh.y, direction="s")
        if msh.gdim > 2:
            self.dydt = self.dudrst(msh.y, direction="t")

        if msh.gdim > 2:
            self.dzdr = self.dudrst(msh.z, direction="r")
            self.dzds = self.dudrst(msh.z, direction="s")
            self.dzdt = self.dudrst(msh.z, direction="t")

        if self.bckend == 'numpy':
            self.drdx = np.zeros_like(self.dxdr, dtype=self.dtype)
            self.drdy = np.zeros_like(self.dxdr, dtype=self.dtype)
            if msh.gdim > 2:
                self.drdz = np.zeros_like(self.dxdr, dtype=self.dtype)

            self.dsdx = np.zeros_like(self.dxdr, dtype=self.dtype)
            self.dsdy = np.zeros_like(self.dxdr, dtype=self.dtype)
            if msh.gdim > 2:
                self.dsdz = np.zeros_like(self.dxdr, dtype=self.dtype)

            if msh.gdim > 2:
                self.dtdx = np.zeros_like(self.dxdr, dtype=self.dtype)
                self.dtdy = np.zeros_like(self.dxdr, dtype=self.dtype)
                self.dtdz = np.zeros_like(self.dxdr, dtype=self.dtype)
        else:
            self.drdx = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)
            self.drdy = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)
            if msh.gdim > 2:
                self.drdz = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)

            self.dsdx = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)
            self.dsdy = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)
            if msh.gdim > 2:
                self.dsdz = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)

            if msh.gdim > 2:
                self.dtdx = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)
                self.dtdy = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)
                self.dtdz = torch.zeros_like(self.dxdr, dtype=self.dtype_d, device=self.device)

        # Find the jacobian determinant, its inverse inverse and mass matrix (3D)
        # jac maps domain from xyz to rst -> dxyz =  jac * drst during integration
        self.log.write(
            "info",
            "Calculating the jacobian determinant and inverse of the jacobian matrix",
        )
        calculate_jacobian_inverse_and_determinant(self)

        # Compute the inverse of the Jacobian determinant
        # self.jac_inv = 1 / self.jac # This is not really used in the way we have it

        # Compute the mass matrix
        self.log.write("info", "Calculating the mass matrix")
        self.B = self.jac * self.w3.reshape((msh.lz, msh.ly, msh.lx))

        # Get area stuff only if mesh is 3D
        # Remember that the area described by two vectors is given by the norm of its cross product
        # i.e., norm(drxds)
        # Here we do that and then multiply by the weights.
        # Similar to what we do with the volume mass matrix.
        # Where we calculate the jacobian determinant
        # and then multiply with weights
        if msh.gdim > 2 and get_area:

            if self.bckend == 'torch':
                raise ValueError("The torch backend does not support facet area calculation yet.")
            
            self.log.write("info", "Calculating area weights and normal vectors")

            self.area = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)
            self.nx = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)
            self.ny = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)
            self.nz = np.zeros((msh.nelv, 6, msh.ly, msh.lx), dtype=self.dtype)

            # ds x dt
            # For facet 1

            d1 = np.stack(
                (self.dxds[:, :, :, 0], self.dyds[:, :, :, 0], self.dzds[:, :, :, 0]),
                axis=3,
            )
            d2 = np.stack(
                (self.dxdt[:, :, :, 0], self.dydt[:, :, :, 0], self.dzdt[:, :, :, 0]),
                axis=3,
            )
            cross = np.cross(d1, d2, axis=3)  # so this is (nelv, k, j, 3)
            norm = np.linalg.norm(cross, axis=3)
            weight = np.outer(self.w, self.w).reshape((1, msh.lz, msh.ly))
            self.area[:, 0, :, :] = norm * weight
            self.nx[:, 0, :, :] = -cross[..., 0] / norm
            self.ny[:, 0, :, :] = -cross[..., 1] / norm
            self.nz[:, 0, :, :] = -cross[..., 2] / norm

            # For facet 2
            d1 = np.stack(
                (
                    self.dxds[:, :, :, -1],
                    self.dyds[:, :, :, -1],
                    self.dzds[:, :, :, -1],
                ),
                axis=3,
            )
            d2 = np.stack(
                (
                    self.dxdt[:, :, :, -1],
                    self.dydt[:, :, :, -1],
                    self.dzdt[:, :, :, -1],
                ),
                axis=3,
            )
            cross = np.cross(d1, d2, axis=3)
            norm = np.linalg.norm(cross, axis=3)
            self.area[:, 1, :, :] = norm * weight
            self.nx[:, 1, :, :] = cross[..., 0] / norm
            self.ny[:, 1, :, :] = cross[..., 1] / norm
            self.nz[:, 1, :, :] = cross[..., 2] / norm

            # dr x dt
            # For facet 3
            d1 = np.stack(
                (self.dxdr[:, :, 0, :], self.dydr[:, :, 0, :], self.dzdr[:, :, 0, :]),
                axis=2,
            )
            d2 = np.stack(
                (self.dxdt[:, :, 0, :], self.dydt[:, :, 0, :], self.dzdt[:, :, 0, :]),
                axis=2,
            )
            cross = np.cross(d1, d2, axis=2).transpose(
                (0, 1, 3, 2)
            )  # Put the result in the last axis
            # After the permutation, the shape is (nelv, k, i, 3)
            norm = np.linalg.norm(cross, axis=3)
            weight = np.outer(self.w, self.w).reshape((1, msh.lz, msh.ly))
            self.area[:, 2, :, :] = norm * weight
            self.nx[:, 2, :, :] = cross[..., 0] / norm
            self.ny[:, 2, :, :] = cross[..., 1] / norm
            self.nz[:, 2, :, :] = cross[..., 2] / norm

            # For facet 4
            d1 = np.stack(
                (
                    self.dxdr[:, :, -1, :],
                    self.dydr[:, :, -1, :],
                    self.dzdr[:, :, -1, :],
                ),
                axis=2,
            )
            d2 = np.stack(
                (
                    self.dxdt[:, :, -1, :],
                    self.dydt[:, :, -1, :],
                    self.dzdt[:, :, -1, :],
                ),
                axis=2,
            )
            cross = np.cross(d1, d2, axis=2).transpose(
                (0, 1, 3, 2)
            )  # Put the result in the last axis
            norm = np.linalg.norm(cross, axis=3)
            weight = np.outer(self.w, self.w).reshape((1, msh.lz, msh.ly))
            self.area[:, 3, :, :] = norm * weight
            self.nx[:, 3, :, :] = -cross[..., 0] / norm
            self.ny[:, 3, :, :] = -cross[..., 1] / norm
            self.nz[:, 3, :, :] = -cross[..., 2] / norm

            # dr x ds
            # For facet 5
            d1 = np.stack(
                (self.dxdr[:, 0, :, :], self.dydr[:, 0, :, :], self.dzdr[:, 0, :, :]),
                axis=1,
            )
            d2 = np.stack(
                (self.dxds[:, 0, :, :], self.dyds[:, 0, :, :], self.dzds[:, 0, :, :]),
                axis=1,
            )
            cross = np.cross(d1, d2, axis=1).transpose(
                (0, 2, 3, 1)
            )  # Put the result in the last axis
            # after the transpose it is (nelv, j, i, 3)
            norm = np.linalg.norm(cross, axis=3)
            weight = np.outer(self.w, self.w).reshape((1, msh.lz, msh.ly))
            self.area[:, 4, :, :] = norm * weight
            self.nx[:, 4, :, :] = -cross[..., 0] / norm
            self.ny[:, 4, :, :] = -cross[..., 1] / norm
            self.nz[:, 4, :, :] = -cross[..., 2] / norm

            # For facet 6
            d1 = np.stack(
                (
                    self.dxdr[:, -1, :, :],
                    self.dydr[:, -1, :, :],
                    self.dzdr[:, -1, :, :],
                ),
                axis=1,
            )
            d2 = np.stack(
                (
                    self.dxds[:, -1, :, :],
                    self.dyds[:, -1, :, :],
                    self.dzds[:, -1, :, :],
                ),
                axis=1,
            )
            cross = np.cross(d1, d2, axis=1).transpose(
                (0, 2, 3, 1)
            )  # Put the result in the last axis
            # after the transpose it is (nelv, j, i, 3)
            norm = np.linalg.norm(cross, axis=3)
            weight = np.outer(self.w, self.w).reshape((1, msh.lz, msh.ly))
            self.area[:, 5, :, :] = norm * weight
            self.nx[:, 5, :, :] = -cross[..., 0] / norm
            self.ny[:, 5, :, :] = -cross[..., 1] / norm
            self.nz[:, 5, :, :] = -cross[..., 2] / norm

        self.log.write("info", "Coef object initialized")
        self.log.write("info", f"Coef data is of type: {self.B.dtype}")
        self.log.toc()

    def dudrst(self, field, direction="r"):
        """
        Perform derivative with respect to reference coordinate r/s/t.

        Used to perform the derivative in the reference coordinates

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).

        direction : str
            Direction to take the derivative. Can be 'r', 's', or 't'. (Default value = 'r').

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.
        """
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        if not self.apply_1d_operators:
            if direction == "r":
                return self.dudrst_3d_operator(field, self.dr)
            elif direction == "s":
                return self.dudrst_3d_operator(field, self.ds)
            elif direction == "t":
                return self.dudrst_3d_operator(field, self.dt)
            else:
                raise ValueError("Invalid direction. Should be 'r', 's', or 't'") 
        elif self.apply_1d_operators:
            if self.bckend == 'numpy':
                if direction == "r":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator(
                            field, self.dr, np.eye(ly, dtype=self.dtype)
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator(
                            field,
                            self.dr,
                            np.eye(ly, dtype=self.dtype),
                            np.eye(lz, dtype=self.dtype),
                        )
                elif direction == "s":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator(
                            field, np.eye(lx, dtype=self.dtype), self.ds
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator(
                            field,
                            np.eye(lx, dtype=self.dtype),
                            self.ds,
                            np.eye(lz, dtype=self.dtype),
                        )
                elif direction == "t":
                    return self.dudrst_1d_operator(
                        field,
                        np.eye(lx, dtype=self.dtype),
                        np.eye(ly, dtype=self.dtype),
                        self.dt,
                    )
                else:
                    raise ValueError("Invalid direction. Should be 'r', 's', or 't'")
            elif self.bckend == 'torch':
                if direction == "r":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator_torch(
                            field, self.dr, torch.eye(ly, dtype=self.dtype_d, device=self.device)
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator_torch(
                            field,
                            self.dr,
                            torch.eye(ly, dtype=self.dtype_d, device=self.device),
                            torch.eye(lz, dtype=self.dtype_d, device=self.device),
                        )
                elif direction == "s":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator_torch(
                            field, torch.eye(lx, dtype=self.dtype_d, device=self.device), self.ds
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator_torch(
                            field,
                            torch.eye(lx, dtype=self.dtype_d, device=self.device),
                            self.ds,
                            torch.eye(lz, dtype=self.dtype_d, device=self.device),
                        )
                elif direction == "t":
                    return self.dudrst_1d_operator_torch(
                        field,
                        torch.eye(lx, dtype=self.dtype_d, device=self.device),
                        torch.eye(ly, dtype=self.dtype_d, device=self.device),
                        self.dt,
                    )
                else:
                    raise ValueError("Invalid direction. Should be 'r', 's', or 't'")
    
    def dudrst_transposed(self, field, direction="r"):
        """
        Perform derivative with respect to reference coordinate r/s/t.

        Used to perform the derivative in the reference coordinates

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).

        direction : str
            Direction to take the derivative. Can be 'r', 's', or 't'. (Default value = 'r').

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.
        """
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        if not self.apply_1d_operators:
            if direction == "r":
                return self.dudrst_3d_operator(field, self.dr.T)
            elif direction == "s":
                return self.dudrst_3d_operator(field, self.ds.T)
            elif direction == "t":
                return self.dudrst_3d_operator(field, self.dt.T)
            else:
                raise ValueError("Invalid direction. Should be 'r', 's', or 't'") 
        elif self.apply_1d_operators:
            if self.bckend == 'numpy':
                if direction == "r":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator(
                            field, self.dr.T, np.eye(ly, dtype=self.dtype)
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator(
                            field,
                            self.dr.T,
                            np.eye(ly, dtype=self.dtype),
                            np.eye(lz, dtype=self.dtype),
                        )
                elif direction == "s":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator(
                            field, np.eye(lx, dtype=self.dtype), self.ds.T
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator(
                            field,
                            np.eye(lx, dtype=self.dtype),
                            self.ds.T,
                            np.eye(lz, dtype=self.dtype),
                        )
                elif direction == "t":
                    return self.dudrst_1d_operator(
                        field,
                        np.eye(lx, dtype=self.dtype),
                        np.eye(ly, dtype=self.dtype),
                        self.dt.T,
                    )
                else:
                    raise ValueError("Invalid direction. Should be 'r', 's', or 't'")
            elif self.bckend == 'torch':
                if direction == "r":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator_torch(
                            field, self.dr.T, torch.eye(ly, dtype=self.dtype_d, device=self.device)
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator_torch(
                            field,
                            self.dr.T,
                            torch.eye(ly, dtype=self.dtype_d, device=self.device),
                            torch.eye(lz, dtype=self.dtype_d, device=self.device),
                        )
                elif direction == "s":
                    if self.gdim == 2:
                        return self.dudrst_1d_operator_torch(
                            field, torch.eye(lx, dtype=self.dtype_d, device=self.device), self.ds
                        )
                    elif self.gdim == 3:
                        return self.dudrst_1d_operator_torch(
                            field,
                            torch.eye(lx, dtype=self.dtype_d, device=self.device),
                            self.ds.T,
                            torch.eye(lz, dtype=self.dtype_d, device=self.device),
                        )
                elif direction == "t":
                    return self.dudrst_1d_operator_torch(
                        field,
                        torch.eye(lx, dtype=self.dtype_d, device=self.device),
                        torch.eye(ly, dtype=self.dtype_d, device=self.device),
                        self.dt.T,
                    )
                else:
                    raise ValueError("Invalid direction. Should be 'r', 's', or 't'")



    def dudrst_3d_operator(self, field, dr):
        """
        Perform derivative with respect to reference coordinate r.

        This method uses derivation matrices from the lagrange polynomials at the GLL points.

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        dr : ndarray
            Derivative matrix in the r/s/t direction to apply to each element. Shape should be (lx*ly*lz, lx*ly*lz).

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.

        Examples
        --------
        Assuming you have a Coef object

        >>> dxdr = coef.dudrst(x, coef.dr)
        """
        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        # ==================================================
        # Using loops
        # dudrst = np.zeros_like(field, dtype=field.dtype)
        # for e in range(0, nelv):
        #    tmp = field[e, :, :, :].reshape(-1, 1)
        #    dtmp = dr @ tmp
        #    dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))
        # ==================================================

        # Using einsum
        field_shape = field.shape
        operator_shape = dr.shape
        field_shape_as_columns = (
            field_shape[0],
            field_shape[1] * field_shape[2] * field_shape[3],
            1,
        )

        # Reshape the field in palce
        field.shape = field_shape_as_columns

        # Calculate the derivative applying the 3D operator broadcasting with einsum
        #dudrst = np.einsum(
        #    "ejk, ekm -> ejm",
        #    dr.reshape(1, operator_shape[0], operator_shape[1]),
        #    field,
        #)
        dudrst = np.matmul(
            dr.reshape(1, operator_shape[0], operator_shape[1]),
            field,
        )

        # Reshape the field back to its original shape
        field.shape = field_shape
        dudrst.shape = field_shape

        return dudrst

    def dudrst_1d_operator(self, field, dr, ds, dt=None):
        """
        Perform derivative applying the 1d operators provided as inputs.

        This method uses the 1D operators to apply the derivative. To apply derivative in r.
        you mush provide the 1d differenciation matrix in that direction and identity in the others.

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        dr : ndarray
            Derivative matrix in the r direction to apply to each element. Shape should be (lx, lx).

        ds : ndarray
            Derivative matrix in the s direction to apply to each element. Shape should be (ly, ly).

        dt : ndarray
            Derivative matrix in the t direction to apply to each element. Shape should be (lz, lz).
            This is optional. If none is passed, it is assumed that the field is 2D.

        Returns
        -------
        ndarray
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.

        Examples
        --------
        Assuming you have a Coef object

        >>> dxdr = coef.dudrst(x, coef.dr, np.eye(ly, dtype=coef.dtype), np.eye(lz, dtype=coef.dtype))
        """
        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        # ==================================================
        # Using loops
        # dudrst = np.zeros_like(field, dtype=field.dtype)
        # for e in range(0, nelv):
        #    tmp = field[e, :, :, :].reshape(-1, 1)
        #    dtmp = apply_1d_operators(tmp, dr, ds, dt, use_broadcast=False)
        #    dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))
        # ==================================================

        # we will use an einsum implementation that expects fields in this form:
        # (nelv, lz, ly, lx) -> (nelv, 1 , lxyz, 1)
        field_shape = field.shape
        operator_shape = dr.shape
        field_shape_as_columns = (
            field_shape[0],
            1,
            field_shape[1] * field_shape[2] * field_shape[3],
            1,
        )

        # Reshape the field in palce
        field.shape = field_shape_as_columns
        # Reshape the operators to comply with that shape
        dr.shape = (1, 1, operator_shape[0], operator_shape[1])
        ds.shape = (1, 1, operator_shape[0], operator_shape[1])
        if not isinstance(dt, type(None)):
            dt.shape = (1, 1, operator_shape[0], operator_shape[1])

        # In the operation, we broadcast in the first two axis.
        # the last two axis are treated as matrices that are multiplied.
        dudrst = apply_1d_operators(field, dr, ds, dt)

        # Reshape everything back to its supposed shape
        field.shape = field_shape
        dr.shape = operator_shape
        ds.shape = operator_shape
        if not isinstance(dt, type(None)):
            dt.shape = operator_shape
        dudrst.shape = field_shape

        return dudrst

    def dudrst_1d_operator_torch(self, field, dr, ds, dt=None):
        """
        Perform derivative applying the 1D operators provided as inputs.

        Parameters
        ----------
        field : torch.Tensor
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        dr : torch.Tensor
            Derivative matrix in the r direction to apply to each element. Shape should be (lx, lx).
        ds : torch.Tensor
            Derivative matrix in the s direction to apply to each element. Shape should be (ly, ly).
        dt : torch.Tensor (optional)
            Derivative matrix in the t direction. Shape should be (lz, lz).

        Returns
        -------
        torch.Tensor
            Derivative of the field with respect to r/s/t. Shape is the same as the input field.
        """
        nelv, lz, ly, lx = field.shape  # Read dimensions

        # Reshape the field for batch-wise matrix multiplication
        field_reshaped = field.view(nelv, 1, lz * ly * lx, 1)

        # Reshape the operators
        dr = dr.view(1, 1, lx, lx)
        ds = ds.view(1, 1, ly, ly)
        if dt is not None:
            dt = dt.view(1, 1, lz, lz)

        # Perform the operation (assumed function to be PyTorch-compatible)
        dudrst = apply_1d_operators_torch(field_reshaped, dr, ds, dt)

        # Reshape everything back
        dudrst = dudrst.view(nelv, lz, ly, lx)

        return dudrst


    def dudxyz(self, field, drdx, dsdx, dtdx=None):
        """
        Perform derivative with respect to physical coordinate x,y,z.

        This method uses the chain rule, first evaluating derivatives with respect to
        rst, then multiplying by the inverse of the jacobian to map to xyz.

        Parameters
        ----------
        field : ndarray
            Field to take derivative of. Shape should be (nelv, lz, ly, lx).
        drdx : ndarray
            Derivative of the reference coordinates with respect to x, i.e.,
            first entry in the appropiate row of the jacobian inverse.
            Shape should be the same as the field.
        dsdx : ndarray
            Derivative of the reference coordinates with respect to y, i.e.,
            second entry in the appropiate row of the jacobian inverse.
            Shape should be the same as the field.
        dtdx : ndarray
            Derivative of the reference coordinates with respect to z, i.e.,
            third entry in the appropiate row of the jacobian inverse.
            Shape should be the same as the field.
            (Default value = None)
            Only valid for 3D fields.

        Returns
        -------
        ndarray
            Derivative of the field with respect to x,y,z. Shape is the same as the input field.

        Examples
        --------
        Assuming you have a Coef object and are working on a 3d field:

        >>> dudx = coef.dudxyz(u, coef.drdx, coef.dsdx, coef.dtdx)
        """

        self.log.write(
            "debug", "Calculating the derivative with respect to physical coordinates"
        )
        self.log.tic()

        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]
        if self.bckend == 'numpy':
            dudxyz = np.zeros_like(field, dtype=field.dtype)
        elif self.bckend == 'torch':
            dudxyz = torch.zeros_like(field, dtype=field.dtype, device=self.device)

        dfdr = self.dudrst(field, direction="r")
        dfds = self.dudrst(field, direction="s")
        if self.gdim > 2:
            dfdt = self.dudrst(field, direction="t")

        # NOTE: DO NOT NEED TO MULTIPLY BY INVERSE OF JACOBIAN DETERMINAT.
        # THIS STEP IS ALREADY DONE IF YOU CALCULATED THE INVERSE WITH NUMPY
        # Here we multiply the derivative in reference element with the respective
        # row of the jacobian

        # ==================================================
        # Using loops
        # if self.gdim > 2:
        #    for e in range(0, nelv):
        #        for k in range(0, lz):
        #            for j in range(0, ly):
        #                for i in range(0, lx):
        #                    # dudxyz[e,k,j,i] = self.jac_inv[e, k, j, i] *
        #                    # (dfdr[e, k, j, i] *drdx[e, k, j, i]  + dfds[e, k, j, i] *
        #                    # dsdx[e, k, j, i]  + dfdt[e, k, j, i] *dtdx[e, k, j, i] )
        #                    dudxyz[e, k, j, i] = (
        #                        dfdr[e, k, j, i] * drdx[e, k, j, i]
        #                        + dfds[e, k, j, i] * dsdx[e, k, j, i]
        #                        + dfdt[e, k, j, i] * dtdx[e, k, j, i]
        #                    )
        # else:
        #    for e in range(0, nelv):
        #        for k in range(0, lz):
        #            for j in range(0, ly):
        #                for i in range(0, lx):
        #                    # dudxyz[e,k,j,i] = self.jac_inv[e, k, j, i] *
        #                    # ( dfdr[e, k, j, i] * drdx[e, k, j, i] +
        #                    # dfds[e, k, j, i] * dsdx[e, k, j, i] )
        #                    dudxyz[e, k, j, i] = (
        #                        dfdr[e, k, j, i] * drdx[e, k, j, i]
        #                        + dfds[e, k, j, i] * dsdx[e, k, j, i]
        #                    )
        # ==================================================

        if self.gdim > 2:
            dudxyz = dfdr * drdx + dfds * dsdx + dfdt * dtdx
        else:
            dudxyz = dfdr * drdx + dfds * dsdx

        self.log.write("debug", "done")
        self.log.toc(level="debug")

        return dudxyz

    def glsum(self, a, comm, dtype=np.double):
        """
        Peform global summatin of given qunaitity a using MPI.

        This method uses MPI to sum over all MPI ranks. It works with any numpy array shape and returns one value.

        Parameters
        ----------
        a : ndarray
            Quantity to sum over all mpiranks.
        comm : Comm
            MPI communicator object.
        datype : numpy.dtype
             (Default value = np.double).

        Returns
        -------
        float
            Sum of the quantity a over all MPI ranks.

        Examples
        --------
        Assuming you have a Coef object and are working on a 3d field:

        >>> volume = coef.glsum(coef.B, comm)
        """

        sendbuf = np.ones((1), dtype)
        if self.bckend == 'numpy':
            sendbuf[0] = np.sum(a)
        elif self.bckend == 'torch':
            sendbuf[0] = torch.sum(a).cpu().numpy()
        recvbuf = np.zeros((1), dtype)
        comm.Allreduce(sendbuf, recvbuf)

        return recvbuf[0]

    def dssum(self, field, msh):
        """
        Peform average of given field over shared points in each rank.

        This method averages the field over shared points in the same rank. It uses the connectivity data in the mesh object.
        dssum might be a missleading name.

        Parameters
        ----------
        field : ndarray
            Field to average over shared points.
        msh : Mesh
            pySEMTools Mesh object.

        Returns
        -------
        ndarray
            Input field with shared points averaged with shared points in the SAME rank.

        Examples
        --------
        Assuming you have a Coef object and are working on a 3d field:

        >>> dudx = coef.dssum(dudx, msh)
        """

        if msh.create_connectivity_bool:

            self.log.write(
                "info", "Averaging field over shared points in the same rank"
            )
            self.log.tic()

            if msh.lz > 1:
                z_ind = [0, msh.lz - 1]
            else:
                z_ind = [0]
            tmp = np.copy(field)

            for e in range(0, msh.nelv):
                # loop through all faces (3 loops required)
                for k in z_ind:
                    for j in range(0, msh.ly):
                        for i in range(0, msh.lx):
                            point = (
                                msh.x[e, k, j, i],
                                msh.y[e, k, j, i],
                                msh.z[e, k, j, i],
                            )
                            point = hash(point)
                            shared_points = msh.coord_hash_to_shared_map[point]
                            shared_points = nonlinear_index(
                                shared_points, msh.lx, msh.ly, msh.lz
                            )
                            field_at_shared = np.array(
                                [
                                    tmp[shared_points[l]]
                                    for l in range(len(shared_points))
                                ]
                            )
                            field[e, k, j, i] = np.mean(field_at_shared)

                for j in [0, msh.ly - 1]:
                    for k in range(msh.lz):
                        for i in range(msh.lx):
                            point = (
                                msh.x[e, k, j, i],
                                msh.y[e, k, j, i],
                                msh.z[e, k, j, i],
                            )
                            point = hash(point)
                            shared_points = msh.coord_hash_to_shared_map[point]
                            shared_points = nonlinear_index(
                                shared_points, msh.lx, msh.ly, msh.lz
                            )
                            field_at_shared = np.array(
                                [
                                    tmp[shared_points[l]]
                                    for l in range(len(shared_points))
                                ]
                            )
                            field[e, k, j, i] = np.mean(field_at_shared)

                for i in [0, msh.lx - 1]:
                    for k in range(msh.lz):
                        for j in range(msh.ly):
                            point = (
                                msh.x[e, k, j, i],
                                msh.y[e, k, j, i],
                                msh.z[e, k, j, i],
                            )
                            point = hash(point)
                            shared_points = msh.coord_hash_to_shared_map[point]
                            shared_points = nonlinear_index(
                                shared_points, msh.lx, msh.ly, msh.lz
                            )
                            field_at_shared = np.array(
                                [
                                    tmp[shared_points[l]]
                                    for l in range(len(shared_points))
                                ]
                            )
                            field[e, k, j, i] = np.mean(field_at_shared)

            self.log.write("info", "done")
            self.log.toc()

        else:
            self.log.write(
                "warning",
                "Mesh does not have connectivity data. Returning unmodified array",
            )

        return field

    def build_spatial_filter(self, r_tf: np.array = None, s_tf: np.array = None, t_tf: np.array = None):
        """
        Build a spatial filter based on the given transfer functions.

        Parameters
        ----------
        r_tf : np.array
            Transfer function for the r dimension. Shape should be (lx, lx).
        s_tf : np.array
            Transfer function for the s dimension. Shape should be (ly, ly).
        t_tf : np.array
            Transfer function for the t dimension. Shape should be (lz, lz).
            This is optional. If none is passed, it is assumed that the field is 2D.

        Returns
        -------
        None
            The spatial filters are stored in the r_filter, s_filter, and t_filter attributes.
        
        """

        if (r_tf is None) or (s_tf is None):
            raise ValueError("Transfer functions must be provided for r and s dimensions.")
        if self.gdim == 3 and t_tf is None:
            raise ValueError("Transfer function for t dimension must be provided for 3D fields.")

        # Build the spatial filter
        self.log.write("info", "Building spatial filter")

        self.r_filter = self.v @ r_tf @ self.vinv
        self.s_filter = self.v @ s_tf @ self.vinv

        if self.bckend == 'torch':
            self.r_filter = torch.as_tensor(self.r_filter, dtype=self.dtype_d, device=self.device)
            self.s_filter = torch.as_tensor(self.s_filter, dtype=self.dtype_d, device=self.device)
        if self.gdim == 3:
            self.t_filter = self.v @ t_tf @ self.vinv
            if self.bckend == 'torch':
                self.t_filter = torch.as_tensor(self.t_filter, dtype=self.dtype_d, device=self.device)
        else:
            self.t_filter = None
        
        self.log.write("info", "1d filters stored in the r_filter, s_filter and t_filter (if 3D) attributes")

    def apply_spatial_filter(self, field):
        """
        Apply the stored spatial filters

        Parameters
        ----------
        field : np.array
            Field to apply the spatial filter to. Shape should be (nelv, lz, ly, lx).

        Returns
        -------
        np.array
            Filtered field. Shape is the same as the input field.

        Notes
        -----
        The spatial filters must be created before calling this function, otherwise, an error will be raised.
        """

        if (not hasattr(self, "r_filter")) or (not hasattr(self, "s_filter")):
            raise ValueError("Spatial filter has not been built. Call build_spatial_filter() first.")

        if self.bckend == 'numpy':
            return self.dudrst_1d_operator(field, self.r_filter, self.s_filter, self.t_filter)
        elif self.bckend == 'torch':
            return self.dudrst_1d_operator_torch(field, self.r_filter, self.s_filter, self.t_filter) 

# -----------------------------------------------------------------------


## Define functions for the calculation of the quadrature points (Taken from the lecture notes)
def GLC_pwts(n, dtype=np.double):
    """Gauss-Lobatto-Chebyshev (GLC) points and weights over [-1,1]

    Parameters
    ----------
    `n` :
        int, number of nodes
        Returns
        `x`: 1D numpy array of size `n`, nodes
        `w`: 1D numpy array of size `n`, weights
    n :


    Returns
    -------


    """

    def delt(i, n):
        """Helper function

        Parameters
        ----------
        i : int

        n : int


        Returns
        -------


        """
        del_ = 1.0
        if i == 0 or i == n - 1:
            del_ = 0.5
        return del_

    x = np.cos(np.arange(n, dtype=dtype) * pi / (n - 1))
    w = np.zeros(n, dtype=dtype)
    for i in range(n):
        tmp_ = 0.0
        for k in range(int((n - 1) / 2)):
            tmp_ += delt(2 * k, n) / (1 - 4.0 * k**2) * np.cos(2 * i * pi * k / (n - 1))
        w[i] = tmp_ * delt(i, n) * 4 / float(n - 1)
    return x, w


def GLL_pwts(n, eps=10**-8, max_iter=1000, dtype=np.double):
    """Generating `n` Gauss-Lobatto-Legendre (GLL) nodes and weights using the
    Newton-Raphson iteration.

    Parameters
    ----------
    `n` :
        int
        Number of GLL nodes
    `eps` :
        float (optional)
        Min error to keep the iteration running
    `maxIter` :
        float (optional)
        Max number of iterations
        Outputs:
    `xi` :
        1D numpy array of size `n`
        GLL nodes
    `w` :
        1D numpy array of size `n`
        GLL weights
        Reference:
        Canuto C., Hussaini M. Y., Quarteroni A., Tang T. A.,
        "Spectral Methods in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
        https://link.springer.com/book/10.1007/978-3-642-84108-8
    n :

    eps :
        (Default value = 10**-8)
    max_iter :
        (Default value = 1000)

    Returns
    -------


    """
    V = np.zeros((n, n), dtype=dtype)  # Legendre Vandermonde Matrix
    # Initial guess for the nodes: GLC points
    xi, _ = GLC_pwts(n, dtype=dtype)
    iter_ = 0
    err = 1000
    xi_old = xi
    while iter_ < max_iter and err > eps:
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
    if iter_ > max_iter and err > eps:
        print("gllPts(): max iterations reached without convergence!")
    # Weights
    w = 2.0 / (n * (n - 1) * V[:, n - 1] ** 2.0)
    return xi, w


def get_transform_matrix(n, dim, apply_1d_operators=False, dtype=np.double):
    """
    get transformation matrix to Legendre space of given order and dimension

    Parameters
    ----------
    n : int
        Polynomial degree (order - 1).

    dim : int
        Dimension of the problem.

    Returns
    -------
    vv : ndarray
        Transformation matrix to Legendre space.
    vvinv : ndarray
        Inverse of the transformation matrix.
    w3 : ndarray
        3D weights.
    x : ndarray
        Quadrature nodes.
    w : ndarray
        Quadrature weights.
    """
    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n, dtype=dtype
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    ww = np.eye((n), dtype=dtype)
    for i in range(0, n):
        ww[i, i] = w[i]

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
    #  The acending rows represent accending polynomial order,
    #  the different columns represent different x_i

    # Allocate space
    leg = np.zeros((p, p), dtype=dtype)
    # First row is filled with 1 according to recursive formula
    leg[0, :] = np.ones((1, p))
    # Second row is filled with x according to recursive formula
    leg[1, :] = np.multiply(np.ones((1, p)), x)

    # Apply the recursive formula for all x_i
    for j in range(1, len(p_v) - 1):
        for k_ in p_v:
            leg[j + 1, k_] = ((2 * j + 1) * x[k_] * leg[j, k_] - j * leg[j - 1, k_]) / (
                j + 1
            )

    leg = leg.T  # nek and I transpose it for transform

    # Scaling factor as in books
    delta = np.ones(n, dtype=dtype)
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
            leg[i, j] = leg[i, j] * delta[j]

    # AA = np.matmul(leg.T, np.matmul(ww, leg))

    # 2d transformation matrix
    v = leg
    vinv = leg.T @ ww
    if not apply_1d_operators:
        v2d = np.kron(v, v)
        vinv2d = np.kron(vinv, vinv)
    else:
        v2d = v
        vinv2d = vinv

    # 3d transformation matrix
    v = leg
    vinv = leg.T @ ww
    if not apply_1d_operators:
        v3d = np.kron(v, np.kron(v, v))
        vinv3d = np.kron(vinv, np.kron(vinv, vinv))
    else:
        v3d = v
        vinv3d = vinv

    if dim == 1:
        vv = v.astype(dtype)
        vvinv = vinv.astype(dtype)
        w3 = w
    elif dim == 2:
        vv = v2d.astype(dtype)
        vvinv = vinv2d.astype(dtype)
        w3 = np.diag(np.kron(ww, ww)).copy()
    else:
        vv = v3d.astype(dtype)
        vvinv = vinv3d.astype(dtype)
        w3 = np.diag(np.kron(ww, np.kron(ww, ww))).copy()

    return vv, vvinv, w3, x, w


def get_derivative_matrix(n, dim, dtype=np.double, apply_1d_operators=False):
    """
    Derivative matrix of Lagrange polynomials a GLL points.

    Parameters
    ----------
    n : int
        Polynomial degree (order - 1).

    dim : int
        Dimension of the problem.

    apply_1d_operators : bool, optional
        If True, the 1D operators will be applied instead of constructing 3d.

    dtype : numpy.dtype, optional
        Data type of the output matrices.

    Returns
    -------
    dx : ndarray
        Derivation matrix wrt r direction.
    dy : ndarray
        Derivation matrix wrt s direction.
    dz : ndarray
        Derivation matrix wrt t direction.
    d_n : ndarray
        Derivation matrix in 1D.
    """
    # Get the quadrature nodes
    x, w_ = GLL_pwts(
        n, dtype=dtype
    )  # The outputs of this functions are not exactly in the order we want (start from 1 not -1)

    # Reorder the quadrature nodes
    x = np.flip(x)
    w = np.flip(w_)

    # Create a diagonal matrix
    ww = np.eye(n, dtype=dtype)
    for i in range(0, n):
        ww[i, i] = w[i]

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
    #  The acending rows represent accending polynomial order,
    #  the different columns represent different x_i

    # Allocate space
    leg = np.zeros((p, p), dtype=dtype)
    # First row is filled with 1 according to recursive formula
    leg[0, :] = np.ones((1, p))
    # Second row is filled with x according to recursive formula
    leg[1, :] = np.multiply(np.ones((1, p)), x)

    # Apply the recursive formula for all x_i
    for j in range(1, len(p_v) - 1):
        for k_ in p_v:
            leg[j + 1, k_] = ((2 * j + 1) * x[k_] * leg[j, k_] - j * leg[j - 1, k_]) / (
                j + 1
            )

    d_n = np.zeros((p, p), dtype=dtype)

    # Simply apply the values as given in the book
    for i in range(0, len(p_v)):
        for j in range(0, len(p_v)):
            if i != j:
                d_n[i, j] = (leg[p - 1, i] / leg[p - 1, j]) * (1 / (x[i] - x[j]))
            if i == 0 and j == 0:
                d_n[i, j] = -(((p - 1) + 1) * (p - 1)) / 4
            if i == (p - 1) and j == (p - 1):
                d_n[i, j] = (((p - 1) + 1) * (p - 1)) / 4
            if i == j and i != 0 and i != (p - 1):
                d_n[i, j] = 0

    if dim == 1:
        dx = d_n
        dy = None
        dz = None
    elif dim == 2:
        if not apply_1d_operators:
            dx2d = np.kron(np.eye(p), d_n)
            dy2d = np.kron(d_n, np.eye(p))
        else:
            dx2d = d_n
            dy2d = d_n

        dx = dx2d
        dy = dy2d
        dz = None
    else:
        if not apply_1d_operators:
            dx3d = np.kron(np.eye(p), np.kron(np.eye(p), d_n))
            dy3d = np.kron(np.eye(p), np.kron(d_n, np.eye(p)))
            dz3d = np.kron(d_n, np.kron(np.eye(p), np.eye(p)))
        else:
            dx3d = d_n
            dy3d = d_n
            dz3d = d_n

        dx = dx3d.astype(dtype)
        dy = dy3d.astype(dtype)
        dz = dz3d.astype(dtype)

    return dx, dy, dz, d_n


def nonlinear_index(linear_index_, lx, ly, lz):
    """
    Map 1d index to 4d

    This is an inverse of linear index.

    Parameters
    ----------
    linear_index_ : list
        List of 1d linear indices.
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.
    Returns
    -------
    list
        List of 4d non linear indices correspoinf to the linear indices.
    """
    indices = []
    for list_ in linear_index_:
        index = np.zeros(4, dtype=int)
        lin_idx = list_
        index[3] = lin_idx / (lx * ly * lz)
        index[2] = (lin_idx - (lx * ly * lz) * index[3]) / (lx * ly)
        index[1] = (lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2]) / lx
        index[0] = (
            lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2] - lx * index[1]
        )
        ind = (index[3], index[2], index[1], index[0])
        indices.append(ind)

    return indices


def apply_1d_operators(x, dr, ds, dt=None, use_broadcast=True):

    if use_broadcast:
        if not isinstance(dt, type(None)):
            return apply_operators_3d(dr, ds, dt, x)
        else:
            return apply_operators_2d(dr, ds, x)
    else:
        if not isinstance(dt, type(None)):
            return apply_operators_3d_no_broadcast(dr, ds, dt, x)
        else:
            return apply_operators_2d_no_broadcast(dr, ds, x)

def apply_1d_operators_torch(x, dr, ds, dt=None):

    if not isinstance(dt, type(None)):
        return apply_operators_3d_torch(dr, ds, dt, x)
    else:
        return apply_operators_2d_torch(dr, ds, x)



def apply_operators_2d_no_broadcast(dr, ds, x):
    """This function applies operators the same way as they are applied in NEK5000
    The only difference is that it is reversed, as this is
    python and we decided to leave that arrays as is"""

    # Apply in r direction
    temp = x.reshape((int(x.size / dr.T.shape[0]), dr.T.shape[0])) @ dr.T

    # Apply in s direction
    temp = ds @ temp.reshape(ds.shape[1], (int(temp.size / ds.shape[1])))

    return temp.reshape(-1, 1)


def apply_operators_3d_no_broadcast(dr, ds, dt, x):
    """This function applies operators the same way as they are applied in NEK5000
    The only difference is that it is reversed, as this is
    python and we decided to leave that arrays as is"""

    # Apply in r direction
    temp = x.reshape((int(x.size / dr.T.shape[0]), dr.T.shape[0])) @ dr.T

    # Apply in s direction
    temp = temp.reshape((ds.shape[1], ds.shape[1], int(temp.size / (ds.shape[1] ** 2))))
    ### The nek5000 way uses a for loop
    ## temp2 = np.zeros((ds.shape[1], ds.shape[0],
    # int(temp.size/(ds.shape[1]**2))))
    # This is needed because dimensions could reduce
    ## for k in range(0, temp.shape[0]):
    ##     temp2[k,:,:] = ds@temp[k,:,:]
    ### We can do it optimized in numpy if we reshape the operator. This way it can broadcast
    temp = ds.reshape((1, ds.shape[0], ds.shape[1])) @ temp

    # Apply in t direction
    temp = dt @ temp.reshape(dt.shape[1], (int(temp.size / dt.shape[1])))

    return temp.reshape(-1, 1)


def apply_operators_2d(dr, ds, x):
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
    #temp = np.einsum("ijkl,ijlm->ijkm", xreshape, drt)
    temp = np.matmul(xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction

    temp = temp.reshape((xshape[0], xshape[1], ds_s1, int(tempsize / ds_s1)))
    #temp = np.einsum("ijkl,ijlm->ijkm", ds, temp)
    temp = np.matmul(ds, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2] * temp.shape[3]

    return temp.reshape((tempshape[0], tempshape[1], tempsize, 1)).copy()

def apply_operators_2d_torch(dr, ds, x):
    """
    This function applies operators the same way as they are applied in NEK5000,
    but optimized for PyTorch.
    """
    dshape = dr.shape
    xshape = x.shape
    xsize = xshape[2] * xshape[3]

    # Transpose the operator in the r direction, leaving the first dimensions unaltered
    drt = dr.transpose(2, 3)
    drt_s0 = drt.shape[2]

    # Reshape the field to be consistent
    xreshape = x.reshape((xshape[0], xshape[1], int(xsize / drt_s0), drt_s0))

    # Apply the operator using einsum
    temp = torch.einsum("ijkl,ijlm->ijkm", xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction
    temp = temp.reshape((xshape[0], xshape[1], ds_s1, int(tempsize / ds_s1)))
    temp = torch.einsum("ijkl,ijlm->ijkm", ds, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2] * temp.shape[3]

    return temp.reshape((tempshape[0], tempshape[1], tempsize, 1)).clone()

def apply_operators_3d(dr, ds, dt, x):
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
    #temp = np.einsum("ijkl,ijlm->ijkm", xreshape, drt)
    temp = np.matmul(xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction
    temp = temp.reshape(
        (xshape[0], xshape[1], ds_s1, ds_s1, int(tempsize / (ds_s1**2)))
    )
    #temp = np.einsum(
    #    "ijklm,ijkmn->ijkln", ds.reshape((dshape[0], dshape[1], 1, ds_s0, ds_s1)), temp
    #)
    temp = np.matmul(ds.reshape((dshape[0], dshape[1], 1, ds_s0, ds_s1)), temp)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3] * temp.shape[4]
    # dt_s0 = dt.shape[2]
    dt_s1 = dt.shape[3]

    # Apply in t direction

    temp = temp.reshape((xshape[0], xshape[1], dt_s1, int(tempsize / dt_s1)))
    #temp = np.einsum("ijkl,ijlm->ijkm", dt, temp)
    temp = np.matmul(dt, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2] * temp.shape[3]

    return temp.reshape((tempshape[0], tempshape[1], tempsize, 1)).copy()

def apply_operators_3d_torch(dr, ds, dt, x):
    """
    This function applies operators the same way as they are applied in NEK5000,
    but optimized for PyTorch.
    """
    dshape = dr.shape
    xshape = x.shape
    xsize = xshape[2] * xshape[3]

    # Transpose the operator in the r direction, leaving the first dimensions unaltered
    drt = dr.transpose(2, 3)
    drt_s0 = drt.shape[2]

    # Reshape the field to be consistent
    xreshape = x.reshape((xshape[0], xshape[1], int(xsize / drt_s0), drt_s0))

    # Apply the operator using einsum
    temp = torch.einsum("ijkl,ijlm->ijkm", xreshape, drt)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3]
    ds_s0 = ds.shape[2]
    ds_s1 = ds.shape[3]

    # Apply in s direction
    temp = temp.reshape((xshape[0], xshape[1], ds_s1, ds_s1, int(tempsize / (ds_s1**2))))
    temp = torch.einsum("ijklm,ijkmn->ijkln", ds.reshape((dshape[0], dshape[1], 1, ds_s0, ds_s1)), temp)

    # Reshape the arrays as needed
    tempsize = temp.shape[2] * temp.shape[3] * temp.shape[4]
    dt_s1 = dt.shape[3]

    # Apply in t direction
    temp = temp.reshape((xshape[0], xshape[1], dt_s1, int(tempsize / dt_s1)))
    temp = torch.einsum("ijkl,ijlm->ijkm", dt, temp)

    # Reshape to proper size
    tempshape = temp.shape
    tempsize = temp.shape[2] * temp.shape[3]

    return temp.reshape((tempshape[0], tempshape[1], tempsize, 1)).clone()



def calculate_jacobian_inverse_and_determinant(self):
    """
    Calculate the inverse of the jacobian matrix and the determinant of the jacobian matrix.

    Parameters
    ----------
    self : Coef
        Coef object.

    Returns
    -------
    None
        Parameters are added into the Coef object
    """

    if self.gdim == 2:
        invert_jacobian_2d(self)
    elif self.gdim == 3:
        invert_jacobian_3d(self)


def invert_jacobian_2d(self):
    """
    Invert the jacobian matrix on all points.

    Here we use the components of the jacobian tensor.
    that is already stored in the object self.

    Parameters
    ----------
    self : Coef
        Coef object.

    Notes
    -----

    If the jacobian matrix is regarded as a tensor, here we use the following notation:

    jac_inv = np.zeros_like(jac)

    a = jac[..., 0, 0]
    b = jac[..., 0, 1]
    c = jac[..., 1, 0]
    d = jac[..., 1, 1]

    det = a * d - b * c

    jac_inv[..., 0, 0] = d / det
    jac_inv[..., 0, 1] = -b / det
    jac_inv[..., 1, 0] = -c / det
    jac_inv[..., 1, 1] = a / det

    Returns
    -------
    None
        Parameters are added into the Coef object
    """

    # Calculate the determinant of the jacobian matrix per point
    self.jac = self.dxdr * self.dyds - self.dxds * self.dydr

    # Get the components of the inverse of the jacobian matrix per point
    self.drdx = self.dyds / self.jac
    self.drdy = -self.dxds / self.jac
    self.dsdx = -self.dydr / self.jac
    self.dsdy = self.dxdr / self.jac

    return


def invert_jacobian_3d(self):
    """
    Invert the jacobian matrix on all points.

    Here we use the components of the jacobian tensor.
    that is already stored in the object self.

    Parameters
    ----------
    self : Coef
        Coef object.

    Notes
    -----

    If the jacobian matrix is regarded as a tensor, here we use the following notation:

    jac_inv = np.zeros_like(jac)

    a = jac[..., 0, 0]
    b = jac[..., 0, 1]
    c = jac[..., 0, 2]
    d = jac[..., 1, 0]
    e = jac[..., 1, 1]
    f = jac[..., 1, 2]
    g = jac[..., 2, 0]
    h = jac[..., 2, 1]
    i = jac[..., 2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    jac_inv[..., 0, 0] = (e * i - f * h) / det
    jac_inv[..., 0, 1] = (c * h - b * i) / det
    jac_inv[..., 0, 2] = (b * f - c * e) / det
    jac_inv[..., 1, 0] = (f * g - d * i) / det
    jac_inv[..., 1, 1] = (a * i - c * g) / det
    jac_inv[..., 1, 2] = (c * d - a * f) / det
    jac_inv[..., 2, 0] = (d * h - e * g) / det
    jac_inv[..., 2, 1] = (b * g - a * h) / det
    jac_inv[..., 2, 2] = (a * e - b * d) / det

    Returns
    -------
    None
        Parameters are added into the Coef object
    """

    # Calculate the determinant of the jacobian matrix per point
    self.jac = (
        self.dxdr * (self.dyds * self.dzdt - self.dydt * self.dzds)
        - self.dxds * (self.dydr * self.dzdt - self.dydt * self.dzdr)
        + self.dxdt * (self.dydr * self.dzds - self.dyds * self.dzdr)
    )

    # Get the components of the inverse of the jacobian matrix per point
    self.drdx = (self.dyds * self.dzdt - self.dydt * self.dzds) / self.jac
    self.drdy = (self.dxdt * self.dzds - self.dxds * self.dzdt) / self.jac
    self.drdz = (self.dxds * self.dydt - self.dxdt * self.dyds) / self.jac
    self.dsdx = (self.dydt * self.dzdr - self.dydr * self.dzdt) / self.jac
    self.dsdy = (self.dxdr * self.dzdt - self.dxdt * self.dzdr) / self.jac
    self.dsdz = (self.dxdt * self.dydr - self.dxdr * self.dydt) / self.jac
    self.dtdx = (self.dydr * self.dzds - self.dyds * self.dzdr) / self.jac
    self.dtdy = (self.dxds * self.dzdr - self.dxdr * self.dzds) / self.jac
    self.dtdz = (self.dxdr * self.dyds - self.dxds * self.dydr) / self.jac

    return

