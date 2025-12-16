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
from pysemtools.solver.linear_solver import LinearSolver
from pysemtools.monitoring import Logger

# -------------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------------
rho = 1.0
Re = 200
nu  = 1/Re   # viscosity
t_end = 200
dt  = 5.0e-3    # as you already choose it
nsteps = int(t_end / dt)

tol_v = 1e-7
tol_p = 1e-4
max_iter_v = 1000
max_iter_p = 1000

output_interval_t = 1

def de_mean_pressure_rhs(q):
    q_mean = coef.glsum(q, comm=comm) / (msh.glb_nelv*msh.lxyz)
    return q - q_mean

# -------------------------------------------------------------
# Read mesh and set up connectivity
# -------------------------------------------------------------
log = Logger(comm=comm, module_name="CylinderFlowSolver")

log.write("info", "---------------------------------------------")
log.write("info", "Cylinder Flow Solver - SEM")
log.write("info", "---------------------------------------------")


log.write("info", "---------------------------------------------")
log.write("info", "Reading mesh and setting up connectivity")
log.write("info", "---------------------------------------------")

msh = Mesh(comm, create_connectivity=False)
fld = FieldRegistry(comm)
pynekread('./../data/cylinder_mesh0.f00000', comm, data_dtype=np.double, msh=msh, fld=fld)
coef = Coef(msh, comm, get_area=False)
conn = MeshConnectivity(comm, msh, use_hashtable=True, max_elem_per_vertex=5, coef=coef)

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
is_cylinder = fld.registry["s0"]==1  # assuming s0 = 1 marks the cylinder

boundary_ids = np.zeros_like(msh.x)
boundary_ids[is_left]   = 1  # left wall
boundary_ids[is_right]  = 2  # right wall
boundary_ids[is_bottom] = 3  # bottom wall
boundary_ids[is_top]    = 4  # top wall
boundary_ids[is_cylinder] = 5  # cylinder wall

# Write out the boundaries to check
fld_ = FieldRegistry(comm)
fld_.add_field(comm, field_name="boundary_ids", field=boundary_ids)
fld_.add_field(comm, field_name="mult", field=conn.multiplicity)
pynekwrite("boundary_ids0.f00000", comm, msh=msh, fld=fld_)
fld_.clear()

def enforce_velocity_bc(u, v):
    # No-slip walls
    u[is_cylinder]   = 0.0; v[is_cylinder]   = 0.0
    
    # Inflow
    u[is_left]    = 1.0; v[is_left]    = 0.0
    
    # Outflow (right - do nothing)
    
    # Top - vertical velocity zero, homogenous neuman (do nothing)
    v[is_top] = 0.0

    # Bottom - vertical velocity zero, homogenous neuman (do nothing)
    v[is_bottom] = 0.0

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
u = np.ones_like(fld.registry['s0']) # Start velocity with 1 everywhere
v = np.zeros_like(fld.registry['s0'])
p = np.zeros_like(fld.registry['s0'])
# initial BCs
u, v = enforce_velocity_bc(u, v)

# AB3 histories for convective term of u and v
rhs_hist_u = [np.zeros_like(u) for _ in range(3)]
rhs_hist_v = [np.zeros_like(v) for _ in range(3)]

# Operators - This build the advection and stiffness operators
## Also initializes the solvers
advection_op  = AdvectionOperator(coef, conn)
stiffness_op  = StiffnessOperator(coef, conn)
solver = LinearSolver(coef, conn, msh)

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

    # Enforce BCs on \hat{v}
    u_hat, v_hat = enforce_velocity_bc(u_hat, v_hat)
 
    # -------------------------------------------------------------
    # Step 2: Pressure Poisson
    # K p^{n+1} = -(1/dt) B div(u_hat)
    # -------------------------------------------------------------
    
    ## Set up the rhs for the poisson eq. for the global domain
    duhat_dx = coef.dudxyz(u_hat, coef.drdx, coef.dsdx)
    dvhat_dy = coef.dudxyz(v_hat, coef.drdy, coef.dsdy)
    div_hat  = duhat_dx + dvhat_dy    
    rhs_p_local = -(1.0/dt) * coef.B * div_hat
    rhs_p = conn.dssum(field=rhs_p_local, msh=msh, average="None")
    ## Compatibility for Neumann Poisson: sum(rhs_p) = 0
    c = coef.glsum(rhs_p, comm=comm) / (msh.glb_nelv*msh.lxyz)
    rhs_p -= c

    ## Set up the pressure preconditioner (only do it once)
    if step == 0:
        jacobi_diag_K = stiffness_op.build_jacobi_diag(msh)
        eps = 1e-14
        Minv_K = 1.0 / (jacobi_diag_K + eps)
        def apply_Minv(r):
            return Minv_K * r

    ## Set up the application of the poisson operator
    def apply_poisson(p_field):
        K_local  = stiffness_op.apply_local(p_field, kappa=1.0)
        K_global = conn.dssum(field=K_local, msh=msh, average="None")
        return K_global

    # Solve for p^{n+1}
    p, res_p, it_p = solver.pcg(apply_poisson, rhs_p, apply_Minv=apply_Minv, x0=p, tol=tol_p, maxiter=max_iter_p, project=de_mean_pressure_rhs)

    # Pressure gauge: make mass-weighted mean(p) = 0 -> this pairs with the compatibility condition step above
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
    
    if step == 0:
        jacobi_diag_K = stiffness_op.build_jacobi_diag(msh)
        b_glob = conn.dssum(field=coef.B, msh=msh, average="None")
        eps = 1e-14
        Minv_v = 1.0 / (b_glob + dt*nu*jacobi_diag_K + eps)
        def apply_Minv_v(r):
            return Minv_v * r

    def apply_helmholtz(phi):
        mass_local = coef.B * phi
        diff_local = stiffness_op.apply_local(phi, kappa=nu)
        A_local    = mass_local + dt * diff_local  # B φ + dt ν K φ
        A_global   = conn.dssum(field=A_local, msh=msh, average="None")
        return A_global

    u, res_u, it_u = solver.pcg(apply_helmholtz, rhs_helm_u, apply_Minv=apply_Minv_v, x0=u, tol=tol_v, maxiter=max_iter_v)
    v, res_v, it_v = solver.pcg(apply_helmholtz, rhs_helm_v, apply_Minv=apply_Minv_v, x0=v, tol=tol_v, maxiter=max_iter_v)

    # Final BC enforcement at t^{n+1}
    u, v = enforce_velocity_bc(u, v)
    
    # -------------------------------------------------------------
    # Step 4: Write output when needed
    # -------------------------------------------------------------

    step_buff += 1
    if step_buff*dt >= output_interval_t:
        file_counter += 1
        step_buff = 0
        fld_ = FieldRegistry(comm)
        fld_.t = (step+1)*dt
        fld_.add_field(comm, field_name="u", field = u, dtype=np.double)
        fld_.add_field(comm, field_name="v", field = v, dtype=np.double)
        fld_.add_field(comm, field_name="p", field = p, dtype=np.double)

        filename = f"field0.f{str(file_counter).zfill(5)}"
        pynekwrite(filename, comm, msh=msh, fld=fld_, istep=step)

    # ------------------------------------------------------------- 
    # Log info about the current step
    # -------------------------------------------------------------
      
    log.write("info", f"Pressure solve:")
    log.write("info", f"    Residual: {res_p:.3e}, iterations: {it_p}")
    log.write("info", f"Velocity solve u:")    
    log.write("info", f"    Residual: {res_u:.3e}, iterations: {it_u}")
    log.write("info", f"Velocity solve v:")
    log.write("info", f"    Residual: {res_v:.3e}, iterations: {it_v}") 

    log.write("info", "---------------------------------------------")







