from cola import Auto, CG, Lanczos, PowerIteration
from functools import partial
from cola.linalg.decompositions.lanczos import lanczos_eigs
import pickle
import time
import gdown
import utils
import datasets
from sampling import EulerMaruyamaPredictor, LangevinCorrector, get_pc_sampler
import mutils
from sde_lib import VPSDE
from configs import get_config
import cola
import flax
import diffrax as dfx
import einops as ein
import equinox as eqx
from jaxtyping import Key, Array, Float32, jaxtyped
import optax
import jax.tree_util as jtu
import jax.random as jr
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import functools as ft
from typing import Optional, Union
from collections.abc import Callable
import math
import ddpm
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from IPython.core.debugger import Pdb

TESTING = True
RETRAIN_MODEL = True
SEED = 42
key = jr.PRNGKey(SEED)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision as tv

# from IPython.display import display, clear_output
# from IPython import display
# torch.manual_seed(3)
# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

spectrum_results = []
output_path = './logs/dense_v2.pkl'


def log_spectrum_results(A, alg, results, output_path):
    out = get_spectrum_results(A, alg=alg)
    results.append(out)
    save_object(results, output_path)


def get_spectrum_results(A, alg):
    tic = time.time()
    if isinstance(alg, cola.Lanczos):
        # TODO: AP to fix this
        raise ValueError("shouldn't use Lanczos")
        eigvals, *_ = lanczos_eigs(A, **alg.__dict__)
        method_name = "lanczos"
    else:
        eigvals, _ = cola.eig(A, k=A.shape[0], alg=alg)
        method_name = "cholesky"
    toc = time.time()
    return {"eigvals": eigvals, "time": toc - tic, "method": method_name}


def save_object(obj, filepath, use_highest=True):
    protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL
    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=protocol)


if not os.path.exists('checkpoint_26'):
    # Replace 'FILE_ID' with the actual file ID
    file_id = '1VZikdcPE2nn8K_da9UUG_JPIzNIRI4yE'

    # Specify the output file name
    output_file = 'checkpoint_26'

    # Download the file
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

if not os.path.exists('checkpoint_199'):
    # Replace 'FILE_ID' with the actual file ID
    file_id = '15VofLMDaxqUKnKwLDzvbDbCpCBm4nUOV'

    # Specify the output file name
    output_file = 'checkpoint_199'

    # Download the file
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

config = get_config()
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
sampling_eps = 1e-3

batch_size = 64
local_batch_size = batch_size // jax.local_device_count()
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0
rng = jax.random.PRNGKey(random_seed)
rng, run_rng = jax.random.split(rng)
rng, model_rng = jax.random.split(rng)
score_model, init_model_state, initial_params = mutils.init_model(run_rng, config)
# optimizer = losses_lib.get_optimizer(config).create(initial_params)
optimizer = None

state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr, model_state=init_model_state,
                     ema_rate=config.model.ema_rate, params_ema=initial_params, rng=rng)  # pytype: disable=wrong-keyword-args
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
state = utils.load_training_state("checkpoint_26", state)


def reshape_tree(tree):
    reshaped_tree = {}
    for k, v in tree.items():
        if isinstance(v, dict) or isinstance(v, flax.core.FrozenDict):
            if 'GroupNorm' in k or 'bn' in k:
                reshaped_tree[k] = v.copy({"scale": v["scale"].reshape(-1), "bias": v["bias"].reshape(-1)})
            else:
                reshaped_tree[k] = reshape_tree(v)
        else:
            reshaped_tree[k] = v
    return reshaped_tree


new_params = reshape_tree(state.params_ema)
# jtu.tree_map(lambda x: x.shape, new_params)
new_state = state.replace(params_ema=new_params)
random_seed = 0
rng = jax.random.PRNGKey(random_seed)
img_size = config.data.image_size
channels = config.data.num_channels
shape = (local_batch_size, img_size, img_size, channels)

rng = jax.random.PRNGKey(random_seed)

rng, step_rng = jax.random.split(rng)
x = sde.prior_sampling(step_rng, shape)
timesteps = jnp.linspace(sde.T, 1e-3, sde.N)
score_fn = mutils.get_score_fn(sde, score_model, new_params, state.model_state, train=False,
                               continuous=config.training.continuous)
rsde = sde.reverse(score_fn, False)

i = 0

# @jax.jit
# def back_step(x, t, dt, rng):
#     rng, step_rng = jax.random.split(rng)
#     z = jr.normal(rng, shape)
#     drift, diffusion = rsde.sde(x, t)
#     x_mean = x + drift * dt
#     x = x_mean + utils.batch_mul(diffusion, jnp.sqrt(-dt) * z)
#     return x, x_mean, rng

# for i in range(sde.N):
#     t = timesteps[i]
#     vec_t = jnp.ones(shape[0]) * t
#     dt = -1.0 / sde.N
#     x, x_mean, rng = back_step(x, vec_t, dt, rng)
#     print(f"{i} / {sde.N}", end="\r")


@jax.jit
def annealed_langevin(x, t, dt, rng):
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    alpha = sde.alphas[timestep]

    std = sde.marginal_prob(x, t)[1]

    def loop_body(step, val):
        rng, x, x_mean = val
        grad = score_fn(x, t)
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, x.shape)
        step_size = (target_snr * std)**2 * 2 * alpha
        x_mean = x + utils.batch_mul(step_size, grad)
        x = x_mean + utils.batch_mul(noise, jnp.sqrt(step_size * 2))
        return rng, x, x_mean

    rng, x, x_mean = jax.lax.fori_loop(0, 10, loop_body, (rng, x, x))
    return x, x_mean, rng


target_snr = 0.16
n_steps = 5000


@jax.jit
def make_step(x_pi, t, key):
    # Calculate step size
    alpha = sde.alphas[t]
    std = sde.marginal_prob(x_pi[None, ...], jnp.ones(1) * t / (sde.N - 1))[1]
    step_size = (target_snr * std)**2 * 2 * alpha

    # Calculate score
    score = score_fn(x_pi[None, ...], jnp.ones(1) * t / (sde.N - 1))

    # Noise for Langevin
    key, eps_key = jr.split(key)
    noise = jr.normal(eps_key, x_pi.shape)[None, ...]

    # Langevin update
    x_mean = x_pi + utils.batch_mul(step_size, score)
    x_pi = x_mean + utils.batch_mul(noise, jnp.sqrt(step_size * 2))
    x_pi = x_pi[0]

    loss = None
    key = jr.split(key)[0]
    return x_pi, key


key = jr.PRNGKey(101)
x_pi = jr.normal(key, (32, 32, 3))
# Setup the plot outside of the loop
fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
image1 = ax1.imshow(inverse_scaler(x_pi))

# display.clear_output(wait=True)
# image1.set_data(inverse_scaler(x_pi))
# display.display(plt.gcf())


class NystromPrecond(cola.ops.LinearOperator):
    """
    Constructs the Nystrom Preconditioner of a linear operator A.

    Args:
        A (LinearOperator): A positive definite linear operator of size (n, n).
        rank (int): The rank of the Nystrom approximation.
        mu (float): Regularization of the linear system (A + mu)x = b.
         Usually, this preconditioner is used to solve linear systems and
         therefore its construction accomodates for the regularization.
        eps (float): Shift used when constructing the preconditioner.
        adjust_mu (bool, optional): Whether to adjust the regularization with the
         estimatted dominant eigenvalue.

    Returns:
        LinearOperator: Nystrom Preconditioner.
    """
    def __init__(self, A, rank, mu=1e-7, eps=1e-8, adjust_mu=True, key=42):
        super().__init__(dtype=A.dtype, shape=A.shape)
        Omega = self.xnp.randn(*(A.shape[0], rank), dtype=A.dtype, device=A.device, key=key)
        self._create_approx(A=A, Omega=Omega, mu=mu, eps=eps, adjust_mu=adjust_mu)

    def _create_approx(self, A, Omega, mu, eps, adjust_mu):
        xnp = self.xnp
        self.Lambda, self.U = get_nys_approx(A=A, Omega=Omega, eps=eps)
        self.adjusted_mu = amu = mu * xnp.max(self.Lambda, axis=0) if adjust_mu else mu
        # Num and denom help for defining inverse and sqrt
        self.subspace_num = xnp.min(self.Lambda) + amu
        self.subspace_denom = self.Lambda + amu
        self.subspace_scaling = self.subspace_num / self.subspace_denom - 1
        self.subspace_scaling = self.subspace_scaling[:, None]
        self.preconditioned_eigmax = xnp.min(self.Lambda) + amu
        self.preconditioned_eigmin = amu

    def _matmat(self, V):
        subspace_term = self.U @ (self.subspace_scaling * (self.U.T @ V))
        return subspace_term + V


def get_nys_approx(A, Omega, eps):
    xnp = A.xnp
    Omega, _ = xnp.qr(Omega, full_matrices=False)
    Y = A @ Omega
    nu = eps * xnp.norm(Y, ord="fro")
    Y += nu * Omega
    C = xnp.cholesky(Omega.T @ Y)
    aux = xnp.solvetri(C, Y.T, lower=True)
    B = aux.T  # shape (params, rank)
    U, Sigma, _ = xnp.svd(B, full_matrices=False)
    Lambda = xnp.clip(Sigma**2.0 - nu, a_min=0.0)
    return Lambda, U


def flat_score_fn(x, t):
    x_img = x.reshape(1, 32, 32, 3)
    score = score_fn(x_img, t * jnp.ones(1))
    return score.reshape(-1)


target_snr = 0.16
n_steps = 2000


def score_hessian(x, t):
    H1 = cola.ops.Jacobian(partial(flat_score_fn, t=t), x)
    return cola.PSD(-(H1.T + H1) / 2)


# @jax.jit


def get_matrices(x, t, key):
    H = score_hessian(x.reshape(-1), t)
    P = cola.ops.I_like(H)  # NystromPrecond(H, rank=30, mu=1e-1, key=key)
    eps = 1e-2 * cola.eigmax(H, alg=PowerIteration(max_iter=5))  # P.adjusted_mu

    # ! ============= ! #
    # ! spectrum step ! #
    log_spectrum_results(H, alg=cola.Eigh(), results=spectrum_results, output_path=output_path)
    # ! spectrum step ! #
    # ! ============= ! #

    reg_H = cola.PSD(H + eps * cola.ops.I_like(H))
    # U = cola.lazify(P.U)
    # D2 = cola.ops.Diagonal(jnp.sqrt(1+P.subspace_scaling[:,0])-1)
    sqrtP = P  # U @ D2 @ U.T + cola.ops.I_like(P)
    # D3 = cola.ops.Diagonal((1+P.subspace_scaling[:,0])**0.25-1)
    P_quart = cola.ops.I_like(P)  # U @ D3 @ U.T + cola.ops.I_like(P)
    inv_H = cola.linalg.inv(reg_H, alg=CG(max_iters=10, P=P))
    # A = cola.PSD(sqrtP @ reg_H @ sqrtP)
    # isqrt_H = P_quart @ cola.linalg.isqrt(A, alg=Lanczos(max_iters=10)) @ P_quart
    isqrt_H = cola.linalg.isqrt(reg_H, alg=Lanczos(max_iters=10))
    return inv_H, isqrt_H


# TODO: yilun's changes in commenting this out
# @jax.jit
def make_step(x_pi, t, key):
    # Calculate step size
    alpha = sde.alphas[t]
    std = sde.marginal_prob(x_pi[None, ...], jnp.ones(1) * t / (sde.N - 1))[1]
    step_size = (target_snr * std)**2 * 2 * alpha

    # Calculate score
    score = score_fn(x_pi[None, ...], jnp.ones(1) * t / (sde.N - 1))

    # Noise for Langevin
    key, eps_key = jr.split(key)
    noise = jr.normal(eps_key, x_pi.shape)[None, ...]

    key, pkey = jr.split(key)
    inv_H, isqrt_H = get_matrices(x_pi, t / (sde.N - 1), pkey)
    # Langevin update
    x_mean = x_pi + utils.batch_mul(step_size, (inv_H @ score.reshape(-1)).reshape(score.shape))
    x_pi = x_mean + utils.batch_mul((isqrt_H @ noise.reshape(-1)).reshape(noise.shape), jnp.sqrt(step_size * 2))
    x_pi = x_pi[0]
    return x_pi, key


# Setup the plot outside of the loop
fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
image1 = ax1.imshow(inverse_scaler(x_pi))

key = jr.PRNGKey(101)
x_pi = jr.normal(key, (32, 32, 3))

# display.clear_output(wait=True)
# image1.set_data(inverse_scaler(x_pi))
# display.display(plt.gcf())

for step in range(n_steps):
    # Setup
    # key, t_key = jr.split(key)
    # t = jr.randint(t_key, (paas,), 30, sde.N-10)
    t = int((2 * jax.nn.sigmoid(-6 * step / n_steps)) * sde.N)
    # t = jnp.ones((paas,), dtype=jnp.int32) * int(((n_steps-step)/n_steps)*sde.N)

    x_pi, key = make_step(x_pi, t, key)
    # print(f"Step {step}, t {t[0]/sde.N:.3f}, Loss {loss:.3f}", end="\r")
    print(f"Step {step}, t {t/sde.N:.3f}", end="\r")

    # if step % 10 == 0 and step:
    #     # plt.imshow(inverse_scaler(jax.vmap(siren)(grid).reshape(img_size, img_size, 3)))
    #     display.clear_output(wait=True)
    #     image1.set_data(inverse_scaler(x_pi))
    #     display.display(plt.gcf())
