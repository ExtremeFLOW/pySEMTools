# Import required modules
import sys
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np

# Get mpi info
comm = MPI.COMM_WORLD
from pysemtools.io.ppymech import pynekwrite, pynekread
from pysemtools.datatypes import Mesh, Coef, FieldRegistry, MeshConnectivity
from pysemtools.solver.operators import AdvectionOperator, StiffnessOperator
from pysemtools.monitoring import Logger

# -------------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------------
rho = 1.0
Re = 3000
nu  = 1/Re   # viscosity
t_end = 300
dt  = 0.25e-2    # as you already choose it
nsteps = int(t_end / dt)

tol_v = 1e-7
tol_p = 1e-4
max_iter_v = 100
max_iter_p = 100

output_interval_t = 1

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

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

        #res_check = np.sqrt(rsnew)/coef.glsum(coef.B, comm=comm)
        res_check = np.sqrt(rsnew)/(msh.glb_nelv*msh.lxyz) # Normalized residual - feels a bit like cheating...

        if res_check < tol:
            #print(f"CG converged in {it} iterations, residual {res_check:.3e}")
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    #print(f"CG finished in {it} iterations, residual {res_check:.3e}")
    return x, res_check, it

# -------------------------------------------------------------
# Read mesh and set up connectivity
# -------------------------------------------------------------
log = Logger(comm=comm, module_name="LidDrivenCavitySolver")

log.write("info", "---------------------------------------------")
log.write("info", "Lid Driven Cavity Solver - SEM")
log.write("info", "---------------------------------------------")


log.write("info", "---------------------------------------------")
log.write("info", "Reading mesh and setting up connectivity")
log.write("info", "---------------------------------------------")

msh = Mesh(comm, create_connectivity=False)
fld = FieldRegistry(comm)
pynekread('./lid_mesh0.f00000', comm, data_dtype=np.double, msh=msh, fld=fld)
coef = Coef(msh, comm, get_area=False)
conn = MeshConnectivity(comm, msh, use_hashtable=True)

# -------------------------------------------------------------
# Identifying the boundaries and setting up boundary conditions
# -------------------------------------------------------------

log.write("info", "---------------------------------------------")
log.write("info", "Setting up boundary conditions")
log.write("info", "---------------------------------------------")

x = msh.x
y = msh.y
x_min, x_max = comm.allreduce(np.min(x), op=MPI.MIN), comm.allreduce(np.max(x), op=MPI.MAX)
y_min, y_max = comm.allreduce(np.min(y), op=MPI.MIN), comm.allreduce(np.max(y), op=MPI.MAX)
eps = 1e-8

is_left   = np.abs(x - x_min) < eps
is_right  = np.abs(x - x_max) < eps
is_bottom = np.abs(y - y_min) < eps
is_top    = np.abs(y - y_max) < eps

lid_speed = 1.0   # U_lid

boundary_ids = np.zeros_like(msh.x)
boundary_ids[is_left]   = 1  # left wall
boundary_ids[is_right]  = 2  # right wall
boundary_ids[is_bottom] = 3  # bottom wall
boundary_ids[is_top]    = 4  # top wall

# Write out the boundaries to check
fld_ = FieldRegistry(comm)
fld_.add_field(comm, field_name="boundary_ids", field=boundary_ids)
pynekwrite("boundary_ids0.f00000", comm, msh=msh, fld=fld_)
fld_.clear()

# Create helper function for the boundary conditions
def enforce_velocity_bc(u, v):
    # Velocity
    u[is_left]   = 0.0; v[is_left]   = 0.0
    u[is_right]  = 0.0; v[is_right]  = 0.0
    u[is_bottom] = 0.0; v[is_bottom] = 0.0
    u[is_top] = lid_speed
    v[is_top] = 0.0

    # Pressure will have homogenous Neumann BCs (do nothing) for now.

    # Make boundary nodes consistent across elements - Consider removing this after checking it works
    u = conn.dssum(field=u, msh=msh, average="multiplicity")
    v = conn.dssum(field=v, msh=msh, average="multiplicity")
    return u, v

# -------------------------------------------------------------
# Set up functions and variables to help in the solution
# -------------------------------------------------------------
log.write("info", "---------------------------------------------")
log.write("info", "Initializing operators")
log.write("info", "---------------------------------------------")

# velocity and pressure Initial conditions in the bulk
u = np.zeros_like(fld.registry['u'])
v = np.zeros_like(fld.registry['u'])
p = np.zeros_like(fld.registry['u'])
# initial BCs
u, v = enforce_velocity_bc(u, v)

# AB3 histories for convective term of u and v
rhs_hist_u = [np.zeros_like(u) for _ in range(3)]
rhs_hist_v = [np.zeros_like(v) for _ in range(3)]

# Operators - This build the advection and stiffness operators
advection_op  = AdvectionOperator(coef, conn)
stiffness_op  = StiffnessOperator(coef, conn)

# -------------------------------------------------------------
# Do the time integration
# -------------------------------------------------------------
log.write("info", "---------------------------------------------")
log.write("info", "Initializing time stepping")
log.write("info", "---------------------------------------------")

file_counter = -1
step_buff = 0
for step in range(nsteps):

    log.write("info", f"tstep: {step}, t={step*dt:.4f}")

    # -------------------------------------------------------------
    # Following deville's book "Numerical Methods for Fluid Dynamics"
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # Step 1: Explicit non linear and forcing terms 
    # -------------------------------------------------------------

    # Make velocities continuous before computing gradients
    u = conn.dssum(field=u, msh=msh, average="multiplicity")
    v = conn.dssum(field=v, msh=msh, average="multiplicity")

    # Nonlinear term N(v^n) = -(u·∇u)B, -(u·∇v)B at time n
    N_u_n = -advection_op.apply_local(u, u, v) # Local, valid element wise
    N_v_n = -advection_op.apply_local(v, u, v)

    # Update AB3 history
    rhs_hist_u[2] = rhs_hist_u[1].copy()
    rhs_hist_u[1] = rhs_hist_u[0].copy()
    rhs_hist_u[0] = N_u_n

    rhs_hist_v[2] = rhs_hist_v[1].copy()
    rhs_hist_v[1] = rhs_hist_v[0].copy()
    rhs_hist_v[0] = N_v_n

    # Current mass-weighted states
    Bu = u * coef.B
    Bv = v * coef.B

    # AB1 / AB2 / AB3 explicit update to get B u_hat, B v_hat
    if step == 0:
        Bu_hat = Bu + dt * rhs_hist_u[0]
        Bv_hat = Bv + dt * rhs_hist_v[0]
    elif step == 1:
        Bu_hat = Bu + dt * (1.5*rhs_hist_u[0] - 0.5*rhs_hist_u[1])
        Bv_hat = Bv + dt * (1.5*rhs_hist_v[0] - 0.5*rhs_hist_v[1])
    else:
        Bu_hat = Bu + dt * (
            (23.0/12.0)*rhs_hist_u[0]
            - (16.0/12.0)*rhs_hist_u[1]
            + (5.0/12.0)*rhs_hist_u[2]
        )
        Bv_hat = Bv + dt * (
            (23.0/12.0)*rhs_hist_v[0]
            - (16.0/12.0)*rhs_hist_v[1]
            + (5.0/12.0)*rhs_hist_v[2]
        )

    # Divide by B to get physical "hat" velocities
    u_hat = Bu_hat / coef.B
    v_hat = Bv_hat / coef.B

    # Enforce lid/wall BCs on \hat{v}
    u_hat, v_hat = enforce_velocity_bc(u_hat, v_hat)

    
    # -------------------------------------------------------------
    # Step 2: Pressure Poisson
    # -------------------------------------------------------------
    # K p^{n+1} = -(1/dt) B div(u_hat)
    duhat_dx = coef.dudxyz(u_hat, coef.drdx, coef.dsdx)
    dvhat_dy = coef.dudxyz(v_hat, coef.drdy, coef.dsdy)
    div_hat  = duhat_dx + dvhat_dy

    # Set up the rhs for the poisson eq. for the global domain
    rhs_p_local = -(1.0/dt) * coef.B * div_hat
    rhs_p = conn.dssum(field=rhs_p_local, msh=msh, average="None")

    # Compatibility for Neumann Poisson: sum(rhs_p) = 0 - This to set up a pressure value
    c = coef.glsum(rhs_p, comm=comm) / (msh.glb_nelv*msh.lxyz)
    rhs_p -= c

    # Poisson operator: K p # Verify that the signs of rhs and lhs are correct. I think they are
    def apply_poisson(p_field):
        K_local  = stiffness_op.apply_local(p_field, kappa=1.0)
        K_global = conn.dssum(field=K_local, msh=msh, average="None")
        return K_global

    # Solve for p^{n+1}
    p, res_p, it_p = cg_solve(apply_poisson, rhs_p, x0=p, tol=tol_p, maxiter=max_iter_p)

    # Pressure gauge: make mass-weighted mean(p) = 0
    p_mean = coef.glsum(p*coef.B, comm=comm) / coef.glsum(coef.B, comm=comm)
    p -= p_mean

    
    # -------------------------------------------------------------
    # Step 2.5: Velocity correction
    # -------------------------------------------------------------
    # u^{tilde} = u_hat - (dt/rho) ∇p
    dp_dx = coef.dudxyz(p, coef.drdx, coef.dsdx)
    dp_dy = coef.dudxyz(p, coef.drdy, coef.dsdy)

    u_tilde = u_hat - (dt/rho) * dp_dx
    v_tilde = v_hat - (dt/rho) * dp_dy

    # Enforce BCs again after correction
    u_tilde, v_tilde = enforce_velocity_bc(u_tilde, v_tilde)

    # -------------------------------------------------------------
    # Step 3: Implicit diffusion solve for u^{n+1}, v^{n+1}
    # -------------------------------------------------------------

    # RHS for Helmholtz (assembled)
    Bu_tilde_local = u_tilde * coef.B
    Bv_tilde_local = v_tilde * coef.B
    rhs_helm_u = conn.dssum(field=Bu_tilde_local, msh=msh, average="None")
    rhs_helm_v = conn.dssum(field=Bv_tilde_local, msh=msh, average="None")

    def apply_helmholtz(phi):
        mass_local = coef.B * phi
        diff_local = stiffness_op.apply_local(phi, kappa=nu)
        A_local    = mass_local + dt * diff_local  # B φ + dt ν K φ
        A_global   = conn.dssum(field=A_local, msh=msh, average="None")
        return A_global

    u, res_u, it_u = cg_solve(apply_helmholtz, rhs_helm_u, x0=u, tol=tol_v, maxiter=max_iter_v)
    v, res_v, it_v = cg_solve(apply_helmholtz, rhs_helm_v, x0=v, tol=tol_v, maxiter=max_iter_v)

    # Final BC enforcement at t^{n+1}
    u, v = enforce_velocity_bc(u, v)

    step_buff += 1
    if step_buff*dt >= output_interval_t:
        file_counter += 1
        step_buff = 0
        fld_ = FieldRegistry(comm)
        fld_.add_field(comm, field_name="u", field = u, dtype=np.double)
        fld_.add_field(comm, field_name="v", field = v, dtype=np.double)
        fld_.add_field(comm, field_name="p", field = p, dtype=np.double)

        vorticuty = coef.dudxyz(v, coef.drdx, coef.dsdx) - coef.dudxyz(u, coef.drdy, coef.dsdy)
        vorticity = conn.dssum(field=vorticuty, msh=msh, average="multiplicity")
        fld_.add_field(comm, field_name="vorticity", field = vorticuty, dtype=np.double)

        from pysemtools.io.ppymech import pynekwrite
        count = step // 10
        filename = f"field0.f{str(file_counter).zfill(5)}"
        pynekwrite(filename, comm, msh=msh, fld=fld_, istep=step)
    
    log.write("info", f"Pressure solve:")
    log.write("info", f"  Residual: {res_p:.3e}, iterations: {it_p}")
    log.write("info", f"Velocity solve u:")
    log.write("info", f"  Residual: {res_u:.3e}, iterations: {it_u}")
    log.write("info", f"Velocity solve v:")
    log.write("info", f"  Residual: {res_v:.3e}, iterations: {it_v}")


    log.write("info", "---------------------------------------------")







