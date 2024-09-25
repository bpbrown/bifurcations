"""
Dedalus script simulating 2.5D horizontally-periodic Rayleigh-Benard convection.

Usage:
    rayleigh_benard_2.5d.py [options]

Options:
    --Nx=<Nx>              Horizontal modes; default is aspect x Nz
    --Nz=<Nz>              Vertical modes [default: 64]
    --aspect=<aspect>      Aspect ratio of domain [default: 4]

    --Rayleigh=<Rayleigh>       Rayleigh number [default: 1e6]

    --run_time_iter=<iter>      How many iterations to run for; defaults to np.inf
    --run_time_simtime=<run>    How long (simtime) to run for [default: 100]

    --label=<label>             Additional label for run output directory
"""
import numpy as np
import dedalus.public as d3
import sys

import logging
logger = logging.getLogger(__name__)

from docopt import docopt
args = docopt(__doc__)

from mpi4py import MPI
nproc = MPI.COMM_WORLD.size

# Parameters
aspect = float(args['--aspect'])
# Parameters
Lx, Lz = aspect, 1
Nz = int(args['--Nz'])
if args['--Nx']:
    Nx = int(args['--Nx'])
else:
    Nx = int(aspect*Nz)

data_dir = './'+sys.argv[0].split('.py')[0]
data_dir += f'_Ra{args["--Rayleigh"]}'
data_dir += f'_Nz{Nz}_Nx{Nx}'
data_dir += f'_a{aspect}'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])
import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')


Rayleigh = float(args['--Rayleigh'])
Prandtl = 1
dealias = 3/2

stop_sim_time = float(args['--run_time_simtime'])
if args['--run_time_iter']:
    stop_iter = int(float(args['--run_time_iter']))
else:
    stop_iter = np.inf

timestepper = d3.SBDF2 #d3.RK222
safety = 0.2
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('y', 'x', 'z', right_handed=False)
dist = d3.Distributor(coords, mesh=[1,nproc], dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_c0 = dist.Field(name='tau_c0')
tau_c1 = dist.Field(name='tau_c1')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
taus = [tau_c0, tau_c1, tau_b1, tau_b2, tau_u1, tau_u2]

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ey, ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)
lift_basis1 = zbasis.derivative_basis(1)
lift1 = lambda A, n: d3.Lift(A, lift_basis1, n)
V = Lx*Lz
volavg = lambda A: d3.integ(A)/V

ω = d3.curl(u)

tau_d = tau_c0 + lift1(tau_c1, -1)
tau_b = lift(tau_b1, -1) + lift(tau_b2, -2)
tau_u = lift(tau_u1, -1) + lift(tau_u2, -2)

# Problem
vars = [p, b, u]
problem = d3.IVP(vars + taus, namespace=locals())
problem.add_equation("div(u) + tau_d = 0")
problem.add_equation("dt(b) - kappa*lap(b) - ez@u + tau_b = - (u@grad(b))")
problem.add_equation("dt(u) - nu*lap(u) + grad(p) - b*ez + tau_u = cross(u, ω)")
problem.add_equation("b(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("integ(ez@tau_u2) = 0")
problem.add_equation("b(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(timestepper, enforce_real_cadence=np.inf)
solver.stop_sim_time = stop_sim_time
solver.stop_iteration = stop_iter

# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
b.low_pass_filter(scales=0.75)
b['g'] *= z * (Lz - z) # Damp noise at walls

snapshots = solver.evaluator.add_file_handler(data_dir+'/snapshots', sim_dt=1, max_writes=20)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(ω, name='vorticity')
snapshots.add_task(d3.div(u), name='divergence')

scalars = solver.evaluator.add_file_handler(data_dir+'/scalars', iter=10)
scalars.add_task(volavg(np.sqrt(u@u)/nu), name='Re')
scalars.add_task(volavg(d3.div(u)), name='div_u')
scalars.add_task(np.sqrt(volavg(d3.div(u)**2)), name='|div_u|')
scalars.add_task(np.sqrt(volavg(tau_d**2)), name='|tau_d|')
scalars.add_task(np.sqrt(volavg(tau_u@tau_u)), name='|tau_u|')
scalars.add_task(np.sqrt(volavg(tau_b**2)), name='|tau_b|')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep,
             cadence=1, safety=safety, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(np.abs(d3.div(u)), name='|div_u|')
flow.add_property(np.sqrt(tau_u@tau_u)+np.sqrt(tau_b**2)+np.sqrt(tau_d**2), name='|taus|')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 1000 == 0:
            for var in vars:
                var['g']
                var['c']
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            max_divu = flow.max('|div_u|')
            max_taus = flow.max('|taus|')
            logger.info(f'Iteration={solver.iteration:d}, Time={solver.sim_time:.2e}, dt={timestep:.2e}, max(Re)={max_Re:.2e}, divu={max_divu:.2e}, taus={max_taus:.2e}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
