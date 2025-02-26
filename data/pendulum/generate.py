"""Generate data of forced damped pendulum.
"""

import argparse
import json
import numpy as np
from scipy.integrate import solve_ivp


def generate_traj(init_cond, omega, gamma, A, f, dt, len_episode):
    def fun(t, s):
        th, thdot = s
        force = A*omega*omega*np.cos(2.0*np.pi*f*t)
        return [thdot, force - gamma*thdot - omega*omega*np.sin(th)]
    sol = solve_ivp(fun, (0.0, dt*(len_episode-1)), init_cond, dense_output=True, method='DOP853')
    t = np.linspace(0.0, dt*(len_episode-1), len_episode)
    return t, sol.sol(t).T


def generate_data(rng, n_samples, range_init, range_omega, range_gamma,
                  range_A, range_f, dt, len_episode, noise_std):
    assert range_init[0] <= range_init[1]
    inits = rng.uniform(low=range_init[0], high=range_init[1], size=n_samples)
    assert range_omega[0] <= range_omega[1]
    omegas = rng.uniform(low=range_omega[0], high=range_omega[1], size=n_samples)
    assert range_gamma[0] <= range_gamma[1]
    gammas = rng.uniform(low=range_gamma[0], high=range_gamma[1], size=n_samples)

    if range_A is None or range_f is None:
        As = np.zeros(n_samples)
        fs = np.zeros(n_samples)
    else:
        assert range_A[0] <= range_A[1]
        assert range_f[0] <= range_f[1]
        As = rng.uniform(low=range_A[0], high=range_A[1], size=n_samples)
        fs = rng.uniform(low=range_f[0], high=range_f[1], size=n_samples)

    # solve ODE
    x = np.empty((n_samples, len_episode))
    for i in range(n_samples):
        t, x_i = generate_traj([inits[i], 0.0], omegas[i], gammas[i], As[i], fs[i], dt, len_episode)
        x[i] = x_i[:,0]

    # observation noise
    x = x + rng.normal(loc=0.0, scale=noise_std, size=x.shape)

    return t, x, inits, omegas, gammas, As, fs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)

    # configurations
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--len-episode', type=int, default=50)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--range-init', type=float, nargs=2, default=[0.0, np.pi/2.0])
    parser.add_argument('--range-omega', type=float, nargs=2, default=[np.pi/4.0, np.pi/2.0])
    parser.add_argument('--range-gamma', type=float, nargs=2, default=[0.0, 0.4])
    parser.add_argument('--range-A', type=float, nargs=2, default=[0.1, 1.0])
    parser.add_argument('--range-f', type=float, nargs=2, default=[np.pi/2.0, np.pi])
    parser.add_argument('--noise-std', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=1234567890)
    args = parser.parse_args()

    # check inputs
    assert args.range_init[0] <= args.range_init[1]
    assert args.range_omega[0] <= args.range_omega[1]
    assert args.range_gamma[0] <= args.range_gamma[1]
    assert args.range_A[0] <= args.range_A[1]
    assert args.range_f[0] <= args.range_f[1]


    # set random seed
    rng = np.random.default_rng(args.seed)


    # generate data
    kwargs = {'range_init': args.range_init,
              'range_omega': args.range_omega, 'range_gamma': args.range_gamma,
              'range_A': args.range_A, 'range_f': args.range_f,
              'dt': args.dt, 'len_episode':args.len_episode, 'noise_std':args.noise_std}
    t, data, inits, omegas, gammas, As, fs = generate_data(rng, args.n_samples, **kwargs)


    # save args
    with open('{}/args_{}.json'.format(args.outdir, args.name), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


    # save data
    np.savetxt('{}/data_{}.txt'.format(args.outdir, args.name), data, fmt='%.9e')
    print('saved data: min(abs(x))={:.3e}, max(abs(x))={:.3e}'.format(
        np.min(np.abs(data)), np.max(np.abs(data)) ))


    # save true parameters
    np.savetxt('{}/true_params_{}.txt'.format(args.outdir, args.name),
        np.stack([inits, omegas, gammas, As, fs], axis=1), fmt='%.9e',
        header='init omega gamma A f')
    print('saved true parameters')
