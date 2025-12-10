# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np

# Get mpi info
comm = MPI.COMM_WORLD

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.solver.operators import AdvectionOperator, StiffnessOperator

msh = Mesh(comm, create_connectivity=True)
fld = FieldRegistry(comm)
pynekread('../data/mixlay0.f00001', comm, data_dtype=np.double, msh=msh, fld=fld)
coef = Coef(msh, comm, get_area=False)
from pysemtools.datatypes.msh_connectivity import MeshConnectivity
msh_conn = MeshConnectivity(comm, msh, rel_tol=1e-5)


# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
min_dx = np.min(coef.dxdr)


dt = min_dx / 1.0 / 10 # time step
#dt = 0.001
print(f"Using time step dt = {dt:.3e}")
kappa = 0.01  # diffusion coefficient

# ---------------------------------------------------------
# Initial condition
# ---------------------------------------------------------
th = np.zeros_like(fld.registry['u'])
u  = np.ones_like(fld.registry['u'])   # constant velocity in x
v  = np.zeros_like(fld.registry['u'])  # zero velocity in y

x0, y0 = 10.0, 7.0      # center of the Gaussian
sigma = 1.0             # width parameter

r2 = (msh.x - x0)**2 + (msh.y - y0)**2
th = np.exp(-r2 / (2.0 * sigma**2))

# Convective RHS history: rhs_hist[0] = N_adv^n, rhs_hist[1] = N_adv^{n-1}, rhs_hist[2] = N_adv^{n-2}
rhs_hist = [np.zeros_like(th) for _ in range(3)]

# Initialize operators

advection_op = AdvectionOperator(coef, msh_conn)
stiffness_op = StiffnessOperator(coef, msh_conn)

# ---------------------------------------------------------
# Conjugate Gradient solver (matrix-free)
# ---------------------------------------------------------
def cg_solve(apply_A, b, x0=None, tol=1e-8, maxiter=200):
    """
    Solve A x = b using Conjugate Gradient.
    apply_A: function that computes A(x).
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    r = b - apply_A(x)
    p = r.copy()
    rsold = np.vdot(r, r).real
    rsold = coef.glsum(rsold, comm=comm)  # global sum

    if rsold == 0.0:
        return x

    for it in range(1, maxiter + 1):
        Ap = apply_A(p)
        pAp = np.vdot(p, Ap).real
        pAp = coef.glsum(pAp, comm=comm)  # global sum
        if pAp == 0.0:
            break

        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.vdot(r, r).real
        rsnew = coef.glsum(rsnew, comm=comm)  # global sum

        res_check = np.sqrt(rsnew)/coef.glsum(coef.B, comm=comm)

        if res_check < tol:
            print(f"CG converged in {it} iterations, residual {res_check:.3e}")
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    print(f"CG finished in {it} iterations, residual {res_check:.3e}")
    return x


# =========================================================
# Time stepping: explicit AB3 for convection, implicit Helmholtz for diffusion
# =========================================================
for step in range(1000):

    coef.log.write("info", f"Step {step}")

    # -------------------------------------------
    # 1) Convective RHS N_adv^n = - (u·∇θ) * B
    # -------------------------------------------
    N_adv_n = -advection_op.apply_local(th, u, v) # Not assembled yet, doing it for the full right hand side only

    # Update convective RHS history (N_adv^n, N_adv^{n-1}, N_adv^{n-2})
    rhs_hist[2] = rhs_hist[1].copy()
    rhs_hist[1] = rhs_hist[0].copy()
    rhs_hist[0] = N_adv_n

    # -------------------------------------------
    # 2) Explicit advection update (AB1/2/3) on B θ
    #    B θ_hat = B θ^n + dt * AB3(N_adv)
    # -------------------------------------------
    Bth = th * coef.B
    if step == 0:
        # 1st-order Euler
        Bth_hat = Bth + dt * rhs_hist[0]
    elif step == 1:
        # 2nd-order Adams–Bashforth (AB2)
        Bth_hat = Bth + dt * (1.5 * rhs_hist[0] - 0.5 * rhs_hist[1])
    else:
        # 3rd-order Adams–Bashforth (AB3)
        Bth_hat = Bth + dt * (
            (23.0 / 12.0) * rhs_hist[0]
            - (16.0 / 12.0) * rhs_hist[1]
            + (5.0  / 12.0) * rhs_hist[2]
        )

    # Assemble RHS: for the helmholtz solve.
    rhs_helm = msh_conn.dssum(field=Bth_hat, msh=msh, average="None")  # sum contributions
    
    
    # -------------------------------------------
    # 3) Implicit diffusion (Helmholtz solve):
    #    (B + dt κ K) θ^{n+1} = B θ_hat
    # -------------------------------------------
    def apply_helmholtz(phi): 
        mass_term = coef.B * phi # phi is the solution at iteration n
        diff_term = stiffness_op.apply_local(phi, kappa=kappa)
        
        return msh_conn.dssum(field=mass_term + dt * diff_term, msh=msh, average="None") # Return the assembled helmholtz operator application

    # Use θ^n as initial guess for CG
    th_new = cg_solve(apply_helmholtz, rhs_helm, x0=th, tol=1e-7, maxiter=50)

    # Update solution
    th = th_new

    # Enforce continuity again after the solve
    th = msh_conn.dssum(field=th, msh=msh, average="multiplicity")

    # -------------------------------------------
    # 4) Diagnostics / plotting
    # -------------------------------------------
    if step % 10 == 0:
        fld_ = FieldRegistry(comm)
        fld_.add_field(comm, field_name="th", field = th, dtype=np.double)

        from pysemtools.io.ppymech import pynekwrite
        count = step // 10
        filename = f"th0.f{str(count).zfill(5)}"
        pynekwrite(filename, comm, msh=msh, fld=fld_, istep=count)