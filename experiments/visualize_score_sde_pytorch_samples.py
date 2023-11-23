import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import ast

sample_storage_path = '/scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/eval/ckpt_26'

for i in range(100):
    samples_i = np.load(os.path.join(sample_storage_path, f'samples_{i}.npz'))

    # save to png
    image_samples_i = Image.fromarray(samples_i['samples'].astype('uint8').squeeze())
    image_samples_i.save(os.path.join(sample_storage_path, f'samples_{i}.png'))

    # with open('/scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/logs/eigs.pkl', 'rb') as file:
    #     loaded_data = pickle.load(file)
    # print(loaded_data)
    # print(f"len(loaded_data)={len(loaded_data)}")

    file_path = '/scratch/yk2516/repos/diffusion_model/second-diffusion/score_sde_pytorch/work_dir/logs/eigs.pkl'

    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read all lines into a list
        lines = file.readlines()

    # Process the lines and convert string representations of lists to actual lists
    for line in lines:
        # Use ast.literal_eval to safely convert the string to a list
        actual_list = ast.literal_eval(line.strip())
        
        # Now you can work with the actual list
        print(actual_list)

    breakpoint()
    plt.figure()
    # for j in range(len(loaded_data)):
    #     plt.plot(loaded_data[j]['eigvals'], label=f'iter_{j}')
    plt.plot(loaded_data['eigvals'], label=f'iter')
    plt.legend()
    plt.show()
    plt.xlabel("eigenvalue indices")
    plt.ylabel("eigenvalues")
    plt.title("Eigen-Spectrum of ∇^2 log p(x)")

    breakpoint()
    plt.savefig(os.path.join(sample_storage_path, f'hessian_eigvals_plot_samples_{i}.png'))


    break



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

