import glob
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import seaborn as sns
from experiments.experiment_fns import extract_datetime
from trainkit.saving import load_object

sns.set(style="whitegrid", font_scale=2.0, rc={"lines.linewidth": 3.0})
sns.set_palette("Set1")

file_pattern = 'eigs_*.pkl'
root_dir = "/home/ubu/Downloads/eig_logs/"
files = glob.glob(file_pattern, root_dir=root_dir)

files_with_dates = [(file, extract_datetime(file)) for file in files]

sorted_files = sorted(files_with_dates, key=lambda x: x[1])

all_eigvals = np.zeros((len(sorted_files), 32 * 32 * 3))
for idx, (file, date) in enumerate(sorted_files):
    print(f"Processing {file}")
    out = load_object(join(root_dir, file))
    all_eigvals[idx] = np.sort(out["eigvals"])[::-1]

time_s = [jdx for jdx in range(all_eigvals.shape[0])]

for time in time_s:
    eigvals = all_eigvals[time]
    plt.figure(dpi=100, figsize=(10, 8))
    plt.title(f"(H.T + H) / 2 | Time: {time} | Cond: {eigvals[0] / eigvals[-1]:1.2f}")
    plt.scatter(np.arange(len(eigvals)), eigvals, label=f"{time}")
    plt.xlabel("Index")
    plt.ylabel("Eigval")
    plt.legend()
    plt.tight_layout()
    plt.show()
