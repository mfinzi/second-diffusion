from jax.random import normal
from jax.random import PRNGKey
import cola
from experiments.experiment_fns import log_spectrum_results

key = PRNGKey(seed=42)
# N = 1000
N = 10
h = normal(key, shape=(N, ))
H = cola.ops.Diagonal(h)
# h = normal(key, shape=(N, N))
# H = cola.ops.Dense(h @ h.T)
H = cola.SelfAdjoint(H)

results = []
output_path = "./logs/eigs.pkl"
log_spectrum_results(H, alg=cola.Eigh(), results=results, output_path=output_path)

max_iters = 10
alg = cola.Lanczos(max_iters=max_iters, tol=1e-3)
log_spectrum_results(H, alg=alg, results=results, output_path=output_path)
