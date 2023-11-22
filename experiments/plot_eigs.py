import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from trainkit.saving import load_object

sns.set(style="whitegrid", font_scale=2.0, rc={"lines.linewidth": 3.0})
sns.set_palette("Set1")

res = load_object('./diffusion/logs/eigs.pkl')

# time_s = [0, 49, 99]
time_s = [0]
for time in time_s:
    eigvals = res[time]["eigvals"]
    eigvals = np.sort(np.array(eigvals))[::-1]

    plt.figure(dpi=100, figsize=(10, 8))
    plt.scatter(np.arange(len(eigvals)), eigvals, label=f"{time}")
    plt.xlabel("Index")
    plt.ylabel("Eigval")
    plt.legend()
    plt.tight_layout()
    plt.show()
