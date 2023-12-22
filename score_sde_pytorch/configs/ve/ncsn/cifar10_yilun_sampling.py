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

# Lint as: python3
"""Config file for reproducing NCSNv1 on CIFAR-10."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = False
  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'none'
  sampling.corrector = 'ald'
  sampling.n_steps_each = 100
  sampling.snr = 0.316

  # evaluation
  evaluate = config.eval
  evaluate.batch_size = 1 # TODO: it was 1024; yilun changed it to 1
  evaluate.enable_sampling = True # TODO: it was False; yilun changed it to True
  evaluate.num_samples = 50 # TODO: it was 50000; yilun changed it to 50
  evaluate.end_ckpt = 16
  
  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.0
  model.embedding_type = 'positional'
  model.conv_size = 3

  # # model
  # model = config.model
  # model.name = 'ncsn'
  # model.scale_by_sigma = False
  # model.sigma_max = 1
  # model.num_scales = 10
  # model.ema_rate = 0.
  # model.normalization = 'InstanceNorm++'
  # model.nonlinearity = 'elu'
  # model.nf = 128
  # model.interpolation = 'bilinear'

  # optim
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 1e-3
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 0
  optim.grad_clip = -1.

  return config
