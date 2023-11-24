import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample_storage_path', help='an example looks like /scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/eval/ckpt_26/2023-11-24/12-42-02')
args = parser.parse_args()
sample_storage_path = args.sample_storage_path

# some config to manually play around
num_samples = 1

for i in range(num_samples):
    samples_i = np.load(os.path.join(sample_storage_path, f'samples_{i}.npz'))

    # save to png
    image_samples_i = Image.fromarray(samples_i['samples'].astype('uint8').squeeze())
    image_samples_i.save(os.path.join(sample_storage_path, f'samples_sde_N1000_{i}.png'))

    # pickle
    # with open('/scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/logs/eigs.pkl', 'rb') as file:
    #     loaded_data = pickle.load(file)
    # print(loaded_data)
    # print(f"len(loaded_data)={len(loaded_data)}")

    # txt
    file_path = os.path.join(sample_storage_path, 'eigs_sde_N1000.txt')
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # plt.figure()
    for j, line in enumerate(lines[i*20:(i+1)*20]):
        actual_eigvals = ast.literal_eval(line.strip())
        sorted_actual_eigvals = sorted(actual_eigvals, key=lambda x: x.real)
        # plt.plot(sorted_actual_eigvals, label=f'iter_{(j+1)*100}')
        ax1.plot(sorted_actual_eigvals, label=f'iter_{(j+1)*100}')


    ax1.legend()
    ax1.set_xlabel("eigenvalue indices (3*32*32=3072 in total)")
    ax1.set_ylabel("eigenvalues")
    ax1.set_title(f"Eigen-Spectrum of ∇^2_x log p(x) for samples {i}")

    # Display the image on the second subplot
    ax2.imshow(image_samples_i)
    ax2.axis('off')  # Turn off axis labels
    ax2.set_title(f'Samples {i}')
    
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(sample_storage_path, f'hessian_eigvals_sde_N1000_plot_samples_{i}.png'))

    # Show the plots
    plt.show()

    # plt.legend()
    # plt.show()
    # plt.xlabel("eigenvalue indices")
    # plt.ylabel("eigenvalues")
    # plt.title(f"Eigen-Spectrum of ∇^2 log p(x) for samples {i}")


    # plt.savefig(os.path.join(sample_storage_path, f'hessian_eigvals_plot_samples_{i}.png'))

    # breakpoint()
    # plt.figure()
    # for j in range(len(loaded_data)):
    #     plt.plot(loaded_data[j]['eigvals'], label=f'iter_{j}')
    # plt.plot(loaded_data['eigvals'], label=f'iter')
    # plt.legend()
    # plt.show()
    # plt.xlabel("eigenvalue indices")
    # plt.ylabel("eigenvalues")
    # plt.title("Eigen-Spectrum of ∇^2 log p(x)")

    # breakpoint()
    # plt.savefig(os.path.join(sample_storage_path, f'hessian_eigvals_plot_samples_{i}.png'))


    



# Load the pickle file
# with open('/scratch/yk2516/repos/diffusion_model/second-diffusion/diffusion/logs/dense.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)

# print(loaded_data)
# print(f"len(loaded_data)={len(loaded_data)}")


# plt.figure()
# for i in range(len(loaded_data)):
#     plt.plot(loaded_data[i]['eigvals'], label=f'iter_{i}')
# plt.legend()
# plt.show()
# plt.xlabel("eigenvalue indices")
# plt.ylabel("eigenvalues")
# plt.title("Eigen-Spectrum of ∇^2 log p(x)")

# plt.savefig('hessian_eigvals_plot.png')

