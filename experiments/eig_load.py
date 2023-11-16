import pickle

# Load the pickle file
with open('/scratch/yk2516/repos/diffusion_model/second-diffusion/experiments/logs/eigs.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)