cd "$HOME/cola"
py -m pip install .[dev]
py -m pip install tensorflow
py -m pip install tensorflow_gan
py -m pip install scipy
py -m pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
py -m pip install matplotlib
py -m pip install ml-collections
