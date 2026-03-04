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

class LinearSolver:
    """Collect iterative solvers for linear systems arising in SEM

    Parameters
    ----------
    coef : Coef
        Coefficient object containing geometric factors and other useful SEM quantities
    conn : MeshConnectivity
        Mesh connectivity object for global assembly
        
    """

    def __init__(self, coef: Coef, conn: MeshConnectivity, msh: Mesh):

        self.coef = coef # This should not be copying the object, just referencing it
        self.conn = conn
        self.msh = msh

    def cg(self, apply_A, b, x0=None, tol=1e-8, maxiter=200, project=None):
        """
        Solve A x = b using Conjugate Gradient.
        apply_A : function that returns A(x)
        project : optional function q -> q_proj (e.g. zero-mean projection for Neumann Poisson)
        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        r = b - apply_A(x)
        if project is not None:
            r = project(r)

        # rsold = (r, r) globally
        rsold = self.coef.glsum(r*r, comm=self.conn.rt.comm)

        if rsold == 0.0:
            return x, 0.0, 0

        p = r.copy()
        N_global = self.msh.glb_nelv * self.msh.lxyz

        for it in range(1, maxiter + 1):
            Ap = apply_A(p)
            pAp = self.coef.glsum(p*Ap, comm=self.conn.rt.comm)
            if pAp == 0.0:
                break

            alpha = rsold / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            if project is not None:
                r = project(r)

            rsnew = self.coef.glsum(r*r, comm=self.conn.rt.comm)

            # RMS residual
            res_check = np.sqrt(rsnew / N_global)

            if res_check < tol:
                break

            p = r + (rsnew / rsold) * p
            rsold = rsnew

        return x, res_check, it

    def pcg(self, apply_A,
                b,
                apply_Minv=None,
                x0=None,
                tol=1e-8,
                maxiter=200,
                project=None):
        """
        Preconditioned Conjugate Gradient:
            A x = b

        Parameters
        ----------
        apply_A : callable
            y = apply_A(x) = A @ x  (matrix-free operator)
        apply_Minv : callable or None
            z = apply_Minv(r) = M^{-1} @ r  (Jacobi, etc).
            If None, no preconditioning (M = I).
        project : callable or None
            q_proj = project(q). Use this to remove nullspace components,
            e.g. de-mean for Neumann Poisson.
        """

        # Initial guess
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        # r0 = b - A x0
        r = b - apply_A(x)
        if project is not None:
            r = project(r)

        # Preconditioned residual z0 = M^{-1} r0
        if apply_Minv is None:
            z = r.copy()
        else:
            z = apply_Minv(r)

        # Dot products (global)
        rz_old = self.coef.glsum(r*z, comm=self.conn.rt.comm)

        rr = self.coef.glsum(r*r, comm=self.conn.rt.comm)

        N_global = self.msh.glb_nelv * self.msh.lxyz

        # Handle trivial case
        if rz_old == 0.0:
            res_check = np.sqrt(rr / N_global)
            return x, res_check, 0

        p = z.copy()

        for it in range(1, maxiter + 1):
            # Ap = A p
            Ap = apply_A(p)

            # pAp = (p, Ap)
            pAp = self.coef.glsum(p*Ap, comm=self.conn.rt.comm)
            if pAp == 0.0:
                break

            alpha = rz_old / pAp

            # x_{k+1} = x_k + alpha p_k
            x = x + alpha * p

            # r_{k+1} = r_k - alpha Ap
            r = r - alpha * Ap
            if project is not None:
                r = project(r)

            # New norms
            rr = self.coef.glsum(r*r, comm=self.conn.rt.comm)

            # RMS residual
            res_check = np.sqrt(rr / N_global)
            if res_check < tol:
                break

            # z_{k+1} = M^{-1} r_{k+1}
            if apply_Minv is None:
                z = r.copy()
            else:
                z = apply_Minv(r)

            rz_new = self.coef.glsum(r*z, comm=self.conn.rt.comm)

            if rz_old == 0.0:
                break

            beta = rz_new / rz_old

            # p_{k+1} = z_{k+1} + beta p_k
            p = z + beta * p
            rz_old = rz_new

        return x, res_check, it