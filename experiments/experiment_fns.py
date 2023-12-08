import time
import gc
import re
import os
from os.path import join
import numpy as np
from datetime import datetime
from trainkit.saving import append_timestamp
from trainkit.saving import save_object
from trainkit.saving import load_object
import cola
from cola.linalg.decompositions.lanczos import lanczos_eigs


def get_results_with_pattern(pattern, dir_path):
    files = get_all_files_with_pattern(pattern, dir_path)
    data = {}
    for file in files:
        iter_n = get_iter(file)
        res = load_object(join(dir_path, file))
        data[iter_n] = res
    return data


def get_iter(file):
    pattern = re.compile(r'\D*(\d+)\.pkl$')
    match = pattern.match(file)
    if match:
        iter_n = match.group(1)
    return int(iter_n)


def get_all_files_with_pattern(pattern, dir_path):
    pattern = re.compile(pattern)
    files = [f for f in os.listdir(dir_path) if pattern.match(f)]
    return files


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


def extract_datetime(filename):
    # date_pattern = r'eigs_(\d{4}-\d{2}-\d{2})_(\d{6})\.pkl'
    date_pattern = r'eigs_(\d{4}-\d{2}-\d{2})_(\d{6})'
    match = re.search(date_pattern, filename)
    if match:
        date_str, time_str = match.groups()
        datetime_str = date_str + time_str
        return datetime.strptime(datetime_str, '%Y-%m-%d%H%M%S')
    return None
