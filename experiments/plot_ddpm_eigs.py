import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from experiments.experiment_fns import get_results_with_pattern

sns.set(style="whitegrid", font_scale=2.0, rc={"lines.linewidth": 3.0})
sns.set_palette("Set1")

# files = get_results_with_pattern("x_", dir_path="./logs/21-09-53/")
res = get_results_with_pattern("H_eigs", dir_path="./logs/09-37-33/")
# res = get_results_with_pattern("HTH_eigs", dir_path="./logs/09-37-33/")
iter_ns = list(res.keys())
iter_ns.sort()

for time in iter_ns:
    eigvals = np.abs(res[time]["eigvals"])
    eigvals = np.sort(eigvals)[::-1]

    plt.figure(dpi=100, figsize=(10, 8))
    plt.scatter(np.arange(len(eigvals)), eigvals, label=f"{time}")
    plt.xlabel("Index")
    plt.ylabel("Eigval")
    plt.legend()
    plt.tight_layout()
    plt.show()
