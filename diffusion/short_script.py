from cola import CG, Lanczos, PowerIteration
# from experiments.experiment_fns import log_spectrum_results
from experiments.experiment_fns import log_from_jax
from functools import partial
import gdown
import utils
import datasets
import mutils
from sde_lib import VPSDE
from configs import get_config
import cola
import flax
import jax.random as jr
import jax.numpy as jnp
import jax
import ddpm
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TESTING = True
RETRAIN_MODEL = True
SEED = 42
key = jr.PRNGKey(SEED)


if not os.path.exists('checkpoint_26'):
    file_id = '1VZikdcPE2nn8K_da9UUG_JPIzNIRI4yE'
    output_file = 'checkpoint_26'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

if not os.path.exists('checkpoint_199'):
    file_id = '15VofLMDaxqUKnKwLDzvbDbCpCBm4nUOV'
    output_file = 'checkpoint_199'
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
                     ema_rate=config.model.ema_rate, params_ema=initial_params, rng=rng)
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

key = jr.PRNGKey(101)
diag = jnp.abs(jr.normal(key, (3072,)))


def score_fn(x, _):
    shape = x.shape
    out = -x.reshape(-1)
    out = diag * out
    return out.reshape(shape)


i = 0
target_snr = 0.16
# n_steps = 2000
n_steps = 100

key = jr.PRNGKey(101)
x_pi = jr.normal(key, (32, 32, 3))
results = []
output_path = "./logs/eigs.pkl"
if not os.path.exists('./logs'):
    os.mkdir('./logs')


def flat_score_fn(x, t):
    x_img = x.reshape(1, 32, 32, 3)
    score = score_fn(x_img, t * jnp.ones(1))
    return score.reshape(-1)


def score_hessian(x, t):
    H1 = cola.ops.Jacobian(partial(flat_score_fn, t=t), x)
    out = -(H1.T + H1) / 2
    # out = H1.T @ H1
    return out


def get_matrices(x, t, key):
    H = score_hessian(x.reshape(-1), t)
    P = cola.ops.I_like(H)
    eps = 1e-2 * cola.eigmax(H, alg=PowerIteration(max_iter=5))

    reg_H = cola.PSD(H + eps * cola.ops.I_like(H))
    # log_spectrum_results(H, Lanczos(max_iters=20, tol=1e-3), results, output_path)
    log_from_jax(reg_H, results, output_path)
    inv_H = cola.linalg.inv(reg_H, alg=CG(max_iters=10, P=P))
    isqrt_H = cola.linalg.isqrt(reg_H, alg=Lanczos(max_iters=10))
    return inv_H, isqrt_H


# @jax.jit
def make_step(x_pi, t, key):
    alpha = sde.alphas[t]
    std = sde.marginal_prob(x_pi[None, ...], jnp.ones(1) * t / (sde.N - 1))[1]
    step_size = (target_snr * std)**2 * 2 * alpha

    score = score_fn(x_pi[None, ...], jnp.ones(1) * t / (sde.N - 1))

    key, eps_key = jr.split(key)
    noise = jr.normal(eps_key, x_pi.shape)[None, ...]

    key, pkey = jr.split(key)
    inv_H, isqrt_H = get_matrices(x_pi, t / (sde.N - 1), pkey)

    x_mean = x_pi + utils.batch_mul(step_size, (inv_H @ score.reshape(-1)).reshape(score.shape))
    x_pi = x_mean + utils.batch_mul((isqrt_H @ noise.reshape(-1)).reshape(noise.shape), jnp.sqrt(step_size * 2))
    x_pi = x_pi[0]
    return x_pi, key


key = jr.PRNGKey(101)
x_pi = jr.normal(key, (32, 32, 3))

for step in range(n_steps):
    t = int((2 * jax.nn.sigmoid(-6 * step / n_steps)) * sde.N)
    x_pi, key = make_step(x_pi, t, key)
    print(f"Step {step}, t {t/sde.N:.3f}", end="\r")
