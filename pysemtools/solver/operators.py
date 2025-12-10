""" Module that contains classes and methods that provide standard usefull quantities in SEM"""

from math import pi
import numpy as np
import sys
if 'torch' in sys.modules:
    import torch
else:
    torch = None
from ..monitoring.logger import Logger
from ..datatypes import Coef, Mesh, MeshConnectivity

#__all__ = ['']

class AdvectionOperator:
    """Build the advection term in SEM 

    Parameters
    ----------
    coef : Coef
        Coefficient object containing geometric factors and other useful SEM quantities

    """

    def __init__(self, coef: Coef, conn: MeshConnectivity):

        self.coef = coef # This should not be copying the object, just referencing it
        self.conn = conn

    def apply_local(self, field: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        """Apply the advection operator to a given field. Local in elements.

        This gives a possitive contribution. Be mindful of signs depending whether this is on the RHS or LHS.

        Parameters
        ----------
        field : np.ndarray
            Field to which the advection operator is applied
        u : np.ndarray
            Advection velocity component in the x (r) direction
        v : np.ndarray
            Advection velocity component in the y (s) direction
        w : np.ndarray, optional
            Advection velocity component in the z (t) direction (for 3D cases), by default None

        Returns
        -------
        np.ndarray
            Resulting field after applying the advection operator
        """

        if self.coef.gdim == 2:
            advection_term_e = u * self.coef.dudxyz(field, self.coef.drdx, self.coef.dsdx)
            advection_term_e += v * self.coef.dudxyz(field, self.coef.drdy, self.coef.dsdy)
        elif self.coef.gdim == 3:
            advection_term_e = u * self.coef.dudxyz(field, self.coef.drdx, self.coef.dsdx, self.coef.dtdx)
            advection_term_e += v * self.coef.dudxyz(field, self.coef.drdy, self.coef.dsdy, self.coef.dtdy)
            advection_term_e += w * self.coef.dudxyz(field, self.coef.drdz, self.coef.dsdz, self.coef.dtdz)

        return advection_term_e * self.coef.B
    
    def apply_global(self, msh: Mesh, field: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        """Apply the advection operator to a given field. Global assembly.

        This gives a possitive contribution. Be mindful of signs depending whether this is on the RHS or LHS.

        Parameters
        ----------
        field : np.ndarray
            Field to which the advection operator is applied
        u : np.ndarray
            Advection velocity component in the x (r) direction
        v : np.ndarray
            Advection velocity component in the y (s) direction
        w : np.ndarray, optional
            Advection velocity component in the z (t) direction (for 3D cases), by default None

        Returns
        -------
        np.ndarray
            Resulting field after applying the advection operator
        """

        advection_local = self.apply_local(field, u, v, w)

        # Assemble globally by summing at shared nodes
        advection_global = self.conn.dssum(field=advection_local, msh=msh, average="None")  # sum, no averaging

        return advection_global
    
class StiffnessOperator:
    """Build the stiffness operator in SEM 

    Parameters
    ----------
    coef : Coef
        Coefficient object containing geometric factors and other useful SEM quantities
    conn : MeshConnectivity
        Mesh connectivity object for global assembly

    """

    def __init__(self, coef: Coef, conn: MeshConnectivity):

        self.coef = coef # This should not be copying the object, just referencing it
        self.conn = conn

        # Calculate the geometric terms and store them in g_ij dictionary
        ## Store the jacobian components in a dictionary
        jac = {}
        jac["11"] = coef.drdx
        jac["12"] = coef.drdy
        if coef.gdim == 3:
            jac["13"] = coef.drdz
        jac["21"] = coef.dsdx
        jac["22"] = coef.dsdy
        if coef.gdim == 3:
            jac["23"] = coef.dsdz    
        if coef.gdim == 3:
            jac["31"] = coef.dtdx
            jac["32"] = coef.dtdy
            jac["33"] = coef.dtdz

        gdim = coef.gdim
        g_ij = {}
        for i in range(gdim):
            for j in range(gdim):
                key = f"{i+1}{j+1}"
                g_ij[key] = np.zeros_like(coef.drdx)
                for k in range(gdim):
                    g_ij[key] += jac[f"{i+1}{k+1}"] * jac[f"{j+1}{k+1}"]
        
        self.g_ij = g_ij

    def apply_local(self, field: np.ndarray, kappa: float = 1.0) -> np.ndarray:
        """Apply the stiffness operator to a given field. Local in elements.

        Parameters
        ----------
        field : np.ndarray
            Field to which the stiffness operator is applied
        kappa : float, optional
            Diffusion coefficient, by default 1.0

        Returns
        -------
        np.ndarray
            Resulting field after applying the stiffness operator
        """

        # 1) Local gradients in reference space
        dudr = self.coef.dudrst(field, direction='r')
        duds = self.coef.dudrst(field, direction='s')
        if self.coef.gdim == 3:
            dudt = self.coef.dudrst(field, direction='t')

        # 2) Geometric terms
        term_r = self.g_ij["11"] * dudr + self.g_ij["12"] * duds
        if self.coef.gdim == 3: term_r += self.g_ij["13"] * dudt

        term_s = self.g_ij["21"] * dudr + self.g_ij["22"] * duds
        if self.coef.gdim == 3: term_s += self.g_ij["23"] * dudt

        if self.coef.gdim == 3:
            term_t = self.g_ij["31"] * dudr + self.g_ij["32"] * duds + self.g_ij["33"] * dudt

        # 3) Multiply by quadrature weights and jacobian determinant (mass)
        term_r *= self.coef.B
        term_s *= self.coef.B
        if self.coef.gdim == 3:
            term_t *= self.coef.B

        # 4) Finish off the term with the weak divergence: D^T (W G grad θ)
        stiff_r_local = self.coef.dudrst_transposed(term_r, direction='r')
        stiff_s_local = self.coef.dudrst_transposed(term_s, direction='s')
        if self.coef.gdim == 3:
            stiff_t_local = self.coef.dudrst_transposed(term_t, direction='t')

        stiff_local = stiff_r_local + stiff_s_local
        if self.coef.gdim == 3:
            stiff_local += stiff_t_local  # element-local contributions

        stiff_local *= kappa

        return stiff_local
    
    def apply_global(self, msh: Mesh, field: np.ndarray, kappa: float = 1.0) -> np.ndarray:
        """Apply the stiffness operator to a given field. Global assembly.

        Parameters
        ----------
        field : np.ndarray
            Field to which the stiffness operator is applied
        kappa : float, optional
            Diffusion coefficient, by default 1.0

        Returns
        -------
        np.ndarray
            Resulting field after applying the stiffness operator
        """

        stiff_local = self.apply_local(field, kappa)

        # 5) Assemble K θ globally by summing at shared nodes
        stiff_global = self.conn.dssum(field=stiff_local, msh=msh, average="None")  # sum, no averaging

        return stiff_global

