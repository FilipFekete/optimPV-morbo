# Diffrax stiff solver version – full option parity with SciPy (Gpulse arg kept for parity),
# supports 'dimentionless' mode and equilibration loop. Returns same shapes as SciPy.
# import jax
# import jax.numpy as jnp
# from jax import random
import warnings,time,re
import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.sparse import lil_matrix
from functools import partial
## Physics constants
from scipy import interpolate, constants
kb = constants.value(u'Boltzmann constant in eV/K')

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import diffrax as dfx
import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

def DBTD_multi_trap_diffrax(parameters, t, Gpulse, t_span, N0=0, G_frac=1, equilibrate=True, eq_limit=1e-2, maxcount=1e3, output_integrated_values=True, **kwargs):
    # Parse parameters (mirror SciPy)
    pnames = [p for p in parameters.keys()]
    if 'k_direct' not in pnames: raise ValueError('k_direct is not in the parameters dictionary')
    if 'L' not in pnames: raise ValueError('L is not in the parameters dictionary')
    if 'alpha' not in pnames: raise ValueError('alpha is not in the parameters dictionary')
    k_direct = float(parameters['k_direct'])
    L = float(parameters['L'])
    alpha = float(parameters['alpha'])

    trapsnames = [p for p in pnames if 'N_t_bulk' in p]
    Cnnames = [p for p in pnames if 'C_n' in p]
    Cpnames = [p for p in pnames if 'C_p' in p]
    Etrapnames = [p for p in pnames if 'E_t_bulk' in p]
    if len(Cpnames) != len(Cnnames):
        for p in pnames:
            if 'ratio_Cnp' in p:
                Cpnames.append('C_p_'+p.split('_')[-1])
    if len(trapsnames) == 0 or len(Etrapnames) == 0 or len(Cnnames) == 0 or len(Cpnames) == 0 or len(trapsnames) != len(Cnnames) or len(trapsnames) != len(Cpnames) or len(trapsnames) != len(Etrapnames):
        raise ValueError('The parameters dictionary must contain at least one trap with its corresponding C_n, C_p and E_trap values')
    for i in range(len(trapsnames)):
        trapsnames[i] = f'N_t_bulk_{i+1}'
        Cnnames[i] = f'C_n_{i+1}'
        Cpnames[i] = f'C_p_{i+1}'
        Etrapnames[i] = f'E_t_bulk_{i+1}'

    N_t_bulk_list = np.asarray([parameters[nm] for nm in trapsnames], dtype=np.float64)
    E_t_bulk_list = np.asarray([parameters[nm] for nm in Etrapnames], dtype=np.float64)
    C_n_bulk_list = np.asarray([parameters[nm] for nm in Cnnames], dtype=np.float64)
    C_p_bulk_list = np.asarray([
        parameters[nm] if nm in parameters else parameters[Cnnames[i]]/parameters['ratio_Cnp_'+str(i+1)]
        for i, nm in enumerate(Cpnames)
    ], dtype=np.float64)

    if 'mu_n' in pnames and 'mu_p' in pnames:
        mu_n = float(parameters['mu_n']); mu_p = float(parameters['mu_p'])
    elif 'mu' in pnames:
        mu_n = float(parameters['mu']); mu_p = float(parameters['mu'])
    else:
        raise ValueError('mu_n and mu_p or mu must be in the parameters dictionary')

    if 'N_c' in pnames and 'N_v' in pnames:
        N_c = float(parameters['N_c']); N_v = float(parameters['N_v'])
    elif 'N_cv' in pnames:
        N_c = float(parameters['N_cv']); N_v = float(parameters['N_cv'])
    else:
        raise ValueError('N_c and N_v or N_cv must be in the parameters dictionary')

    if 'Eg' not in pnames: raise ValueError('Eg must be in the parameters dictionary')
    Eg = float(parameters['Eg'])
    T = float(parameters.get('T', 300.0))

    # kwargs (mirror names, including misspelling)
    dimentionless = kwargs.get('dimentionless', False)  # keep the misspelling for API parity
    grid_size = int(kwargs.get('grid_size', 100))
    rtol = float(kwargs.get('rtol', 1e-3))
    atol = float(kwargs.get('atol', 1e-6))
    method = kwargs.get('method', 'LSODA')  # accepted but ignored, parity only
    use_jacobian = kwargs.get('use_jacobian', False)   # accepted but ignored
    timeout = float(kwargs.get('timeout', 60))
    timeout_solve = float(kwargs.get('timeout_solve', 60))
    dt0 = kwargs.get('dt0', None)
    max_steps = int(kwargs.get('max_steps', 100_000))

    # Derived quantities
    ni = np.sqrt(N_c*N_v*np.exp(-Eg/(kb*T)))
    p1s = N_v*np.exp(-E_t_bulk_list/(2*kb*T))
    n1s = N_c*np.exp((E_t_bulk_list-Eg)/(kb*T))
    D_n = mu_n * kb * T
    D_p = mu_p * kb * T
    ft = (C_n_bulk_list*ni + C_p_bulk_list*p1s)/(C_n_bulk_list*(ni + n1s) + C_p_bulk_list*(p1s + ni))

    number_of_traps = len(N_t_bulk_list)
    z_array = np.linspace(0.0, L, grid_size, dtype=np.float64)
    dz = z_array[1] - z_array[0]

    # Generation profile (precomputed for parity; not used in RHS to match SciPy code paths)
    mean_beer_lambert = np.mean(np.exp(-alpha * z_array))
    generation = np.zeros((len(t_span), len(z_array)), dtype=np.float64)
    for i in range(len(t_span)):
        generation[i] = Gpulse[i] * np.exp(-alpha * z_array) / mean_beer_lambert

    # Initial conditions (same as SciPy)
    N_init = N0 * G_frac
    n0_z = N_init * np.exp(-alpha * z_array) / np.mean(np.exp(-alpha * z_array))

    P_init = np.zeros((number_of_traps+2, grid_size), dtype=np.float64)
    for j in range(number_of_traps+2):
        if j < 2:
            P_init[j, :] = n0_z
        else:
            P_init[j, :] = N_t_bulk_list[j-2]*ft[j-2]
    P0_np = P_init.reshape(-1)

    # JAX constants (64-bit)
    k_direct_j = jnp.asarray(k_direct, dtype=jnp.float64)
    Eg_j = jnp.asarray(Eg, dtype=jnp.float64)
    Bulk_tr = jnp.asarray(N_t_bulk_list, dtype=jnp.float64)
    Bn = jnp.asarray(C_n_bulk_list, dtype=jnp.float64)
    Bp = jnp.asarray(C_p_bulk_list, dtype=jnp.float64)
    ETrap = jnp.asarray(E_t_bulk_list, dtype=jnp.float64)
    Nc = jnp.asarray(N_c, dtype=jnp.float64)
    Nv = jnp.asarray(N_v, dtype=jnp.float64)
    Tj = jnp.asarray(T, dtype=jnp.float64)
    Dnj = jnp.asarray(D_n, dtype=jnp.float64)
    Dpj = jnp.asarray(D_p, dtype=jnp.float64)
    dzj = jnp.asarray(dz, dtype=jnp.float64)

    # Optional dimensionless rescaling
    tau = None
    if dimentionless:
        # scale initial conditions
        P0_np = P0_np / ni
        n0_z = n0_z / ni
        # time scale: tau based on traps
        taus = 1.0/(N_t_bulk_list * np.sqrt(C_n_bulk_list * C_p_bulk_list))
        taus = taus[~np.isinf(taus)]
        tau = float(np.average(taus)) if len(taus) > 0 else 1.0
        # scale times and diffusion
        t_span = np.asarray(t_span, dtype=np.float64) / tau
        D_n = D_n * tau/(L**2)
        D_p = D_p * tau/(L**2)
        # scale generation (kept for parity but not used)
        generation = generation * (tau / ni)
        # update JAX constants for dimensionless model
        Dnj = jnp.asarray(D_n, dtype=jnp.float64)
        Dpj = jnp.asarray(D_p, dtype=jnp.float64)

    # Laplacian with Neumann zero-flux boundaries
    def second_derivative_jax(P):
        interior = (P[2:] - 2.0*P[1:-1] + P[:-2]) / (dzj*dzj)
        left = 2.0*(P[1] - P[0]) / (dzj*dzj)
        right = 2.0*(P[-2] - P[-1]) / (dzj*dzj)
        return jnp.concatenate([left[None], interior, right[None]])

    # RHS for dimensional model

    def rhs_dim(ti, P_flat, _):
        P = P_flat.reshape((number_of_traps+2, grid_size))
        n = jnp.clip(P[0], 0.0, jnp.inf)
        p = jnp.clip(P[1], 0.0, jnp.inf)
        ntr = jnp.clip(P[2:], 0.0, jnp.inf)

        kT = kb * Tj
        ni2 = Nc*Nv*jnp.exp(-Eg_j/kT)

        e_capture = Bn[:, None] * n[None, :] * (Bulk_tr[:, None] - ntr)
        h_capture = Bp[:, None] * p[None, :] * ntr
        e_emission = (Nc * jnp.exp(-(Eg_j - ETrap) / kT) * Bn)[:, None] * ntr
        h_emission = (Nv * jnp.exp(-ETrap / kT) * Bp)[:, None] * (Bulk_tr[:, None] - ntr)

        d2n = second_derivative_jax(n)
        d2p = second_derivative_jax(p)

        dn = - k_direct_j * (n * p - ni2) - jnp.sum(e_capture, axis=0) + jnp.sum(e_emission, axis=0) + Dnj * d2n
        dp = - k_direct_j * (n * p - ni2) - jnp.sum(h_capture, axis=0) + jnp.sum(h_emission, axis=0) + Dpj * d2p
        dtr_list = e_capture - e_emission - h_capture + h_emission
        return jnp.concatenate([dn[None, :], dp[None, :], dtr_list]).reshape(-1)

    # RHS for dimensionless model

    def rhs_dimless(ti, P_flat, _):
        P = P_flat.reshape((number_of_traps+2, grid_size))
        n = jnp.clip(P[0], 0.0, jnp.inf)
        p = jnp.clip(P[1], 0.0, jnp.inf)
        ntr = jnp.clip(P[2:], 0.0, jnp.inf)

        kT = kb * Tj
        e_capture = Bn[:, None] * n[None, :] * ( (Bulk_tr/ni)[:, None] - ntr)
        h_capture = Bp[:, None] * p[None, :] * ntr
        e_emission = ((Nc/ni) * jnp.exp(-(Eg_j - ETrap) / kT) * Bn)[:, None] * ntr
        h_emission = ((Nv/ni) * jnp.exp(-ETrap / kT) * Bp)[:, None] * ( (Bulk_tr/ni)[:, None] - ntr)

        d2n = second_derivative_jax(n)
        d2p = second_derivative_jax(p)

        # Note: in dimless model, ni^2 -> 1 and Nc,Nv are scaled by ni already
        dn = - (k_direct_j * ni * tau) * (n * p - 1.0) - jnp.sum(e_capture, axis=0) + jnp.sum(e_emission, axis=0) + Dnj * d2n
        dp = - (k_direct_j * ni * tau) * (n * p - 1.0) - jnp.sum(h_capture, axis=0) + jnp.sum(h_emission, axis=0) + Dpj * d2p
        dtr_list = e_capture - e_emission - h_capture + h_emission
        return jnp.concatenate([dn[None, :], dp[None, :], dtr_list]).reshape(-1)

    # Choose model
    term = dfx.ODETerm(rhs_dimless if dimentionless else rhs_dim)
    solver = dfx.Kvaerno5()
    stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)

    # Initial state
    P0 = jnp.asarray(P0_np, dtype=jnp.float64)

    # Equilibration loop: repeatedly integrate over t_span and reinject n0_z until convergence
    if equilibrate:
        start_global = time.time()
        count = 0
        end_point = jnp.zeros(grid_size) + 1e-20
        while True:
            if (time.time() - start_global) > timeout:
                return np.nan*np.ones((len(t),grid_size)), np.nan*np.ones((len(t),grid_size))
            start_single = time.time()
            sol_single = dfx.diffeqsolve(
                term,
                solver,
                t0=float(t_span[0] if not dimentionless else (t_span[0])),
                t1=float(t_span[-1] if not dimentionless else (t_span[-1])),
                dt0=dt0,
                y0=P0,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
                saveat=dfx.SaveAt(ts=jnp.asarray(t_span, dtype=jnp.float64)),
                throw=False,
            )
            if (time.time() - start_single) > timeout_solve or sol_single.result != dfx.RESULTS.successful:
                return np.nan*np.ones((len(t),grid_size)), np.nan*np.ones((len(t),grid_size))

            Y = sol_single.ys  # (len(t_span), nvars)
            P_end = Y[-1].reshape((number_of_traps+2, grid_size))
            n_last = P_end[0]
            p_last = P_end[1]

            # Inject new carriers
            inj = jnp.asarray(n0_z, dtype=jnp.float64 if not dimentionless else jnp.float64)
            n_next = n_last + inj
            p_next = p_last + inj
            P0 = jnp.concatenate([n_next, p_next, P_end[2:].reshape(-1)])

            new_end = n_last
            RealChange = jnp.abs((new_end - end_point) / end_point)
            end_point = new_end
            count += 1
            if np.asarray(jnp.all(RealChange < eq_limit)) or count > maxcount:
                if count > maxcount:
                    return np.nan*np.ones((len(t),grid_size)), np.nan*np.ones((len(t),grid_size))
                break

    # Final simulation on t
    if dimentionless and (tau is not None) and tau != 0:
        t_eval = jnp.asarray(np.asarray(t, dtype=np.float64) / tau, dtype=jnp.float64)
    else:
        t_eval = jnp.asarray(t, dtype=jnp.float64)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=float(t_eval[0]),
        t1=float(t_eval[-1]),
        dt0=dt0,
        y0=P0,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=dfx.SaveAt(ts=t_eval),
        throw=False,
    )

    if sol.result != dfx.RESULTS.successful:
        return np.nan*np.ones((len(t),grid_size)), np.nan*np.ones((len(t),grid_size))

    Y = np.asarray(sol.ys)
    sol_flat = Y.reshape((len(t), number_of_traps+2, grid_size))
    n_dens = sol_flat[:, 0, :]
    p_dens = sol_flat[:, 1, :]

    if dimentionless:
        n_dens = n_dens * ni
        p_dens = p_dens * ni

    if output_integrated_values:
        n_dens_mid = (n_dens[:, 1:] + n_dens[:, :-1]) / 2
        p_dens_mid = (p_dens[:, 1:] + p_dens[:, :-1]) / 2
        n_list = [n_dens_mid[i] for i in range(len(t))]
        p_list = [p_dens_mid[i] for i in range(len(t))]
        return n_list, p_list
    else:
        return n_dens, p_dens