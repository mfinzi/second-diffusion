py -m pip install --upgrade pip
py -m pip install jupyter
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd "$HOME/cola"
py -m pip install .[dev]
cd "$HOME/second-diffusion"
py -m pip install tensorflow
py -m pip install tensorflow_datasets
py -m pip install tensorflow_gan
py -m pip install scipy
py -m pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
py -m pip install matplotlib
py -m pip install ml-collections
