import time
import pickle
import cola
from cola.linalg.decompositions.lanczos import lanczos_eigs


def log_from_jax(A, results, output_path):
    tic = time.time()
    xnp = A.xnp
    eigvals, _ = xnp.eigh(A.to_dense())
    toc = time.time()
    out = {"eigvals": eigvals, "time": toc - tic, "method": "cholesky"}
    results.append(out)
    save_object(results, output_path)
    print("*=" * 50 + "\nSaved\n" + "*=" * 50)


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


def save_object(obj, filepath, use_highest=True):
    protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL
    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=obj, file=f, protocol=protocol)
