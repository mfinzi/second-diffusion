import time
import gc
import numpy as np
from trainkit.saving import append_timestamp
from trainkit.saving import save_object
import cola
from cola.linalg.decompositions.lanczos import lanczos_eigs


def log_from_jax(A, output_path):
    tic = time.time()
    xnp = A.xnp
    # eigvals = xnp.sum(A.to_dense())
    eigvals, _ = xnp.eigh(A.to_dense())
    # eigvals = xnp.zeros(A.shape, device=A.device, dtype=A.dtype)
    toc = time.time()
    out = {"eigvals": np.array(eigvals), "time": toc - tic, "method": "cholesky"}
    save_object(out, append_timestamp(output_path, True))
    del eigvals
    del out
    print("*=" * 50 + "\nSaved\n" + "*=" * 50)
    gc.collect()


def log_spectrum_results(A, alg, results, output_path):
    out = get_spectrum_results(A, alg=alg)
    results.append(out)
    save_object(results, output_path)


def get_spectrum_results(A, alg):
    tic = time.time()
    if isinstance(alg, cola.Lanczos):
        # TODO: AP to fix this
        eigvals, *_ = lanczos_eigs(A, **alg.__dict__)
        method_name = "lanczos"
    else:
        eigvals, _ = cola.eig(A, k=A.shape[0], alg=alg)
        method_name = "cholesky"
    toc = time.time()
    return {"eigvals": eigvals, "time": toc - tic, "method": method_name}
