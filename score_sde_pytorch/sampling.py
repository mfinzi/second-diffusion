# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils

from functools import partial
import cola
import time
import pickle

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, workdir=None, use_pred_cond=False):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device,
                                 workdir=workdir,
                                 use_pred_cond=use_pred_cond)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, workdir=None, use_pred_cond=False):
    super().__init__(sde, score_fn, probability_flow)
    self.workdir = workdir
    self.use_pred_cond = use_pred_cond

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False, workdir=None, use_pred_cond=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

    self.workdir = workdir
    self.use_pred_cond = use_pred_cond

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)

    print(f"(1000-timestep.item())={(1000-timestep.item())} for hessian computation", flush=True)
    tic = time.time()
    H = torch.autograd.functional.jacobian(partial(self.flat_score_fn,t=t), x.reshape(-1)) # it takes 26 seconds to finish this
    H_eigvals, _ = torch.linalg.eig(H)
    toc = time.time()

    output_hessian_computation = {"eigvals": np.array(H_eigvals.cpu()), "timestep": timestep.item(), "time": toc - tic, "method": "torch.linalg.eigvals"}

    use_highest = True
    filepath = f"{self.workdir}/eigs_sde_N{sde.N}.pkl"
    filepath_txt = f"{self.workdir}/eigs_sde_N{sde.N}.txt"
    filepath_time_txt = f'{self.workdir}/eigs_sde_N{sde.N}_time.txt'

    protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL

    with open(file=filepath, mode='wb') as f:
        pickle.dump(obj=output_hessian_computation, file=f, protocol=protocol)
    with open(filepath_txt, 'a') as f_filepath_txt:
        f_filepath_txt.write(str(output_hessian_computation['eigvals'].tolist()) + '\n')  
    with open(filepath_time_txt, 'a') as f_filepath_time_txt:
        f_filepath_time_txt.write(str(output_hessian_computation['time']) + '\n')

    if self.use_pred_cond:
      P_mat = H.T @ H
    else:
      P_mat = torch.eye(x.reshape(-1).shape[0])
    
    # breakpoint()
    
    noise = torch.randn_like(x)

    if self.use_pred_cond:
      soln = torch.linalg.solve(P_mat, score.view(-1))
      P_mat_eigvals, P_mat_eigvecs = torch.linalg.eigh(P_mat)
      soln_sqrt = P_mat_eigvecs @ torch.diag(1/torch.sqrt(P_mat_eigvals)) @ P_mat_eigvecs.T
      x_mean = (x + beta[:, None, None, None] * soln.reshape(x.shape)) / torch.sqrt(1. - beta)[:, None, None, None]
      x = x_mean + torch.sqrt(beta)[:, None, None, None] * (soln_sqrt @ noise.view(-1)).reshape(x.shape)
    else:
      x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
      x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise

    # Computing Hessian Eigenvalues (Every 100 Iterations)
    # if ((1000-timestep.item()) % 200) == 0 or (1000-timestep.item()) in [925, 950, 975, 990, 993, 995, 996, 997, 998, 999]:
      # print(f"(1000-timestep.item())={(1000-timestep.item())} for hessian computation", flush=True)
      # tic = time.time()
      # H = torch.autograd.functional.jacobian(partial(self.flat_score_fn,t=t), x.reshape(-1)) # it takes 26 seconds to finish this

      # H_eigvals = torch.linalg.eigvals(H) # take 4 seconds to finish this
      # toc = time.time()
      
      # output_hessian_computation = {"eigvals": np.array(H_eigvals.cpu()), "timestep": timestep.item(), "time": toc - tic, "method": "torch.linalg.eigvals"}

      # use_highest = True
      # filepath = f"{self.workdir}/eigs_sde_N{sde.N}.pkl"
      # filepath_txt = f"{self.workdir}/eigs_sde_N{sde.N}.txt"
      # filepath_time_txt = f'{self.workdir}/eigs_sde_N{sde.N}_time.txt'

      # protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL

      # with open(file=filepath, mode='wb') as f:
      #     pickle.dump(obj=output_hessian_computation, file=f, protocol=protocol)
      # with open(filepath_txt, 'a') as f_filepath_txt:
      #     f_filepath_txt.write(str(output_hessian_computation['eigvals'].tolist()) + '\n')  
      # with open(filepath_time_txt, 'a') as f_filepath_time_txt:
      #     f_filepath_time_txt.write(str(output_hessian_computation['time']) + '\n')

      # # TODO: work in progress  
      # H1 = cola.ops.Jacobian(partial(self.flat_score_fn,t=t), x)

    return x, x_mean

  def flat_score_fn(self, x, t):
    x = x.reshape(1, 3, 32, 32)
    return self.score_fn(x, t).reshape(-1)

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False, workdir=None, use_pred_cond=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps, workdir = None, use_pred_cond = False):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    
    self.workdir = workdir
    self.use_pred_cond = use_pred_cond

  def flat_score_fn(self, x, t):
    x = x.reshape(1, 3, 32, 32)
    return self.score_fn(x, t).reshape(-1)

  def compute_hessian_eigvals(self, x, t, i):
    # time step to log hessian eigenvalues
    log_hessian_eigval_conditions = (i % 100 == 0) or (i in [925, 950, 975, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999])

    tic = time.time()
    if self.use_pred_cond or log_hessian_eigval_conditions:
      print(f"i={i} for hessian computation", flush=True)
      H = torch.autograd.functional.jacobian(partial(self.flat_score_fn,t=t), x.reshape(-1))
    else:
      H = None

    if log_hessian_eigval_conditions:
      print(f"i={i} for hessian eigvals computation", flush=True)
      H_eigvals = torch.linalg.eigvals(H)
    toc = time.time()

    if log_hessian_eigval_conditions:
      output_hessian_computation = {"eigvals": np.array(H_eigvals.cpu()), "timestep_i": i, "time": toc - tic, "method": "torch.linalg.eigvals"}

      use_highest = True
      filepath = f"{self.workdir}/eigs_LangevinCorrector.pkl"
      filepath_txt = f"{self.workdir}/eigs_LangevinCorrector.txt"
      filepath_time_txt = f'{self.workdir}/eigs_time_LangevinCorrector.txt'

      protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL

      with open(file=filepath, mode='wb') as f:
          pickle.dump(obj=output_hessian_computation, file=f, protocol=protocol)
      with open(filepath_txt, 'a') as f_filepath_txt:
          f_filepath_txt.write(str(output_hessian_computation['eigvals'].tolist()) + '\n')  
      with open(filepath_time_txt, 'a') as f_filepath_time_txt:
          f_filepath_time_txt.write(str(output_hessian_computation['time']) + '\n')

    return H

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      H = self.compute_hessian_eigvals(x.float(), t, i)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      
      if self.use_pred_cond:
        P_mat = H.T @ H
      else:
        P_mat = torch.eye(x.reshape(-1).shape[0])

      # breakpoint()

      if self.use_pred_cond:
        soln = torch.linalg.solve(P_mat.double(), grad.view(-1))
        P_mat_eigvals, P_mat_eigvecs = torch.linalg.eigh(P_mat)
        soln_sqrt = P_mat_eigvecs @ torch.diag(1/torch.sqrt(P_mat_eigvals)) @ P_mat_eigvecs.T
        
        x_mean = x + step_size[:, None, None, None] * soln.reshape(x.shape)
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * (soln_sqrt @ noise.view(-1)).reshape(x.shape)
      else:
        x_mean = x + step_size[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps, workdir=None, use_pred_cond=False):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    self.workdir = workdir
    self.use_pred_cond = use_pred_cond

  def flat_score_fn(self, x, t):
    x = x.reshape(1, 3, 32, 32)
    return self.score_fn(x.float(), t).reshape(-1)

  def compute_hessian_eigvals(self, x, t, i):
    # time step to log hessian eigenvalues
    # log_hessian_eigval_conditions = ((1000-1000*t) % 100 == 0) or ((1000-1000*t) in [925, 950, 975, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999])
    log_hessian_eigval_conditions = (t.item() < 0.01)
    
    tic = time.time()
    if self.use_pred_cond or log_hessian_eigval_conditions:
      print(f"t={t}, i={i} for hessian computation", flush=True)
      H = torch.autograd.functional.jacobian(partial(self.flat_score_fn,t=t), x.reshape(-1))
    else:
      H = None

    if log_hessian_eigval_conditions:
      print(f"t={t}, i={i} for hessian eigvals computation", flush=True)
      H_eigvals = torch.linalg.eigvals(H)
    toc = time.time()

    if log_hessian_eigval_conditions:
      output_hessian_computation = {"eigvals": np.array(H_eigvals.cpu()), "timestep_t": t, "timestep_i": i, "time": toc - tic, "method": "torch.linalg.eigvals"}

      use_highest = True
      filepath = f"{self.workdir}/eigs_AnnealedLangevinDynamics.pkl"
      filepath_txt = f"{self.workdir}/eigs_AnnealedLangevinDynamics.txt"
      filepath_time_txt = f'{self.workdir}/eigs_time_AnnealedLangevinDynamics.txt'

      protocol = pickle.HIGHEST_PROTOCOL if use_highest else pickle.DEFAULT_PROTOCOL

      with open(file=filepath, mode='wb') as f:
          pickle.dump(obj=output_hessian_computation, file=f, protocol=protocol)
      with open(filepath_txt, 'a') as f_filepath_txt:
          f_filepath_txt.write(str(output_hessian_computation['eigvals'].tolist()) + '\n')  
      with open(filepath_time_txt, 'a') as f_filepath_time_txt:
          f_filepath_time_txt.write(str(output_hessian_computation['time']) + '\n')

    return H

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x.float(), t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]
    
    log_hessian_eigval_conditions = (t.item() < 0.01)

    if log_hessian_eigval_conditions:
      H = self.compute_hessian_eigvals(x, t, i)

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps, workdir=None, use_pred_cond=False):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous, workdir=None, use_pred_cond=False):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow, workdir=workdir, use_pred_cond=use_pred_cond)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow, workdir=workdir, use_pred_cond=use_pred_cond)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps, workdir=None, use_pred_cond=False):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps, workdir=workdir, use_pred_cond=use_pred_cond)
  return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', workdir=None, use_pred_cond=False):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous,
                                          workdir=workdir,
                                          use_pred_cond=use_pred_cond)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps,
                                          workdir=workdir,
                                          use_pred_cond=use_pred_cond)

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t

        # x = x.float()
        # vec_t = vec_t.float()

        # breakpoint()
        x = x.float()
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x = x.float()
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
        # breakpoint()
      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      return x, nfe

  return ode_sampler
