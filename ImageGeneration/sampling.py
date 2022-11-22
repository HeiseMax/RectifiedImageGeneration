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

import matplotlib.pyplot as plt

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


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
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
  elif sampler_name.lower() == 'ode_step':
    sampling_fn = get_ode_sampler_with_step(sde=sde,
                                  shape=shape,
                                  n_meta_steps=config.sampling.sample_N,
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
                                 device=config.device)
  elif sampler_name.lower() == 'pc_step':
    sde.N=config.sampling.sample_N
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
                                 device=config.device)
  elif sampler_name.lower() == 'mixup':
    sampling_fn = get_mixup_sampler(sde=sde, shape=shape, inverse_scaler=inverse_scaler, device=config.device)
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
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

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
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

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
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

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
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
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
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

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
      #x = 0.0 * sde.prior_sampling(shape).to(device)
      print(x.max(), x.min())
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
      
      print('N STEPS:', sde.N, 'NFE:', sde.N * (n_steps + 1), denoise, x_mean.min(), x_mean.max())
      #import torchvision
      #x_img = inverse_scaler(x_mean) 
      #print(x_img.min(), x_img.max())
      #torchvision.utils.save_image(x_img.clamp(0. , 1.), 'figs/subvp_ode.png', nrow=8, normalize=True) 
      #assert False

      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler

def get_pc_sampler_with_step(sde, shape, n_meta_steps, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
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
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

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
      timesteps = torch.linspace(sde.T, eps, n_meta_steps, device=device)

      for i in range(n_meta_steps):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
      
      print('N STEPS:', n_meta_steps, 'NFE:', n_meta_steps * (n_steps + 1), x_mean.min(), x_mean.max())

      return inverse_scaler(x_mean if denoise else x), n_meta_steps * (n_steps + 1)

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
      
      print('init:', x.shape, x.min(), x.max())

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
      #print('end:', denoise, x.shape, x.min(), x.max(), nfe)
      print('ODE:', 'NFE:', nfe, x.min(), x.max())

      return x, nfe

  return ode_sampler


def get_ode_sampler_with_step(sde, shape, n_meta_steps, inverse_scaler,
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
        #x = sde.prior_sampling(shape).to(device)
        
        z0 = np.load('assets/afhq_cat_init.npy')
        x =  torch.tensor(z0, device=device)[1:2] 
      else:
        x = z
      
      timesteps = torch.linspace(sde.T, eps, n_meta_steps, device=device)
      dt = (sde.T - eps)/n_meta_steps 
      
      #x_cllt = []
      #x_cllt.append(x[0:1].detach().clone().cpu())
      #print(sde.T, eps)
      
      for i in range(n_meta_steps):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        x_pred = x - t * drift
        x = (x - dt * drift).detach().clone()
        import torchvision
        torchvision.utils.save_image(torch.clamp(x, -1.0, 1.0), 'figs/subvp_real_sample_%d.png'%i, nrow=1, normalize=True) 
        torchvision.utils.save_image(torch.clamp(x_pred, -1.0, 1.0), 'figs/subvp_target_sample_%d.png'%i, nrow=1, normalize=True) 
        #x_cllt.append(x[0:1].detach().clone().cpu())
      
      ''' 
      #t_list = [0, 1,2,3,4, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
      t_list = [0,1,2,5,10,15,20,25,50,75,100,300,400, 500,600, 700, 800, 900, 999]
      #dt_list = []
      for i in range(len(t_list)):
        
        import torchvision
        torchvision.utils.save_image(inverse_scaler(x_cllt[t_list[i]]).clamp_(0.0, 1.0), 'figs/real_%d.png'%t_list[i], nrow=1, normalize=False) 

        t = torch.ones(shape[0], device=device) * timesteps[t_list[i]]
        print(t)
        drift = drift_fn(model, x_cllt[t_list[i]].to(device), t)
        x_pred = x_cllt[t_list[i]].to(device) - t[0].item() * drift
        torchvision.utils.save_image(inverse_scaler(x_pred).clamp_(0.0, 1.0), 'figs/pred_%d.png'%t_list[i], nrow=1, normalize=False)
      assert False 
      '''
      
      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      x = inverse_scaler(x)
      print('ODE: N STEPS:', n_meta_steps, 'NFE:', n_meta_steps, x.min(), x.max())
      
      #import torchvision
      #torchvision.utils.save_image(x.clamp_(0.0, 1.0), 'figs/subvp_ode.png', nrow=8, normalize=True) 
      assert False
      
      return x, n_meta_steps

  return ode_sampler

def get_mixup_sampler(sde, shape, inverse_scaler, device='cuda'):
  """
  Get mixup flow sampler

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  def mixup_sampler(model, z=None):
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
        ### VP and subVP are standard Gaussian, VE is standard Gaussian * sigma_max
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device) #* 50.
        x = z0.detach().clone()

        #z0 = np.load('assets/rematch_data_reflow_org/rematch_fake_z0_ckpt_fid_257_seed_0.npy')
        #z0 = np.load('assets/afhq_cat_init.npy')
        #z0 = np.load('assets/afhq_cat_random_init.npy')
        #print(z0.shape)
        #indices = torch.randperm(z0.shape[0])[:shape[0]]
        #z0 = z0[indices]
        #x = torch.tensor(z0).to(device)[:shape[0]].float()
        #x = torch.tensor(z0).to(device)[2:3].float()
        #x = x.repeat(shape[0], 1, 1, 1)
        #print(x.shape, sde.sample_N)
        
        #data = np.load('assets/rematch_data/rematch_fake_data_ckpt_fid_257_seed_0.npy') 
        #data = data[indices]
        #data = torch.tensor(data).to(device)
        #import torchvision
        #torchvision.utils.save_image(data, 'figs/data.png', nrow=10, normalize=True)
        #assert False

        #import torchvision
        #torchvision.utils.save_image(x, 'figs/init.png', nrow=1, normalize=True) 

        ### NOTE: interpolation
        '''   
        #z0 = np.load('assets/afhq_cat_init.npy')
        z0 = np.load('assets/afhq_cat_random_init.npy')
        x0 = torch.tensor(z0, device=device)[27:28]
        x1 = torch.tensor(z0, device=device)[37:38]
        interp = np.linspace(0., np.pi/2., shape[0])
        interp = torch.tensor(interp, device=device)
        interp = torch.sin(interp)
        interp = interp.view(-1, 1, 1, 1).repeat(1, shape[1], shape[2], shape[3])
        x = x0 * interp + x1 * torch.sqrt(1.-interp.pow(2)) 
        x = x.float()
        '''

        ### NOTE: latent encoding
        #z0 = np.load('assets/cat_img.npy')
        #z0 = np.load('assets/cat_init.npy')
        #z0 = np.load('assets/afhq_cat_imgs.npy')[58:59]
        #x = torch.tensor(z0, device=device)
        #import torchvision
        #torchvision.utils.save_image(x, 'figs/init.png', nrow=1, normalize=True)
        
        ### NOTE: SeFa
        '''
        z0 = np.load('assets/rematch_data_reflow_org/rematch_fake_z0_ckpt_fid_257_seed_0.npy')
        x = torch.tensor(z0[0:1]).to(device).float()
        x = x.repeat(shape[0], 1, 1, 1)
        eigen = np.load('assets/cifar10_eigen/eigen_vectors.npy')
        eigen = torch.tensor(eigen[:, 3]).to(device).view(1, shape[1], shape[2], shape[3])
        print(eigen.shape, x.shape, eigen.norm())
        alpha = np.linspace(-2., 2., shape[0])
        for i in range(shape[0]):
            x[i] = x[i] + alpha[i] * eigen
            print(alpha[i], x[i].norm())
        '''

      else:
        x = z
      
      '''
      ### NOTE: for conditional generation 
      ### ====begin====
      z0 = x.detach().clone()
      mask = torch.zeros_like(z0, device=device)
      mask[:, :, 12:20, 12:20] = 1.0
      #target = torch.ones_like(z0, device=device) * (-1.)
      real_data = np.load('/scratch/cluster/xcliu/improved-precision-and-recall-metric/data/cifar10.npy')
      target = torch.tensor(real_data)
      target = target[4:5].repeat(shape[0], 1, 1, 1).to(device)
      print(target.shape, target.min(), target.max())
      import torchvision
      torchvision.utils.save_image(target[:64], 'figs/data.png', nrow=8, normalize=True) 
      #assert False
      ### ====end====      
      '''

      #x_cllt = []
      #x_cllt.append(x[0:1].detach().clone().cpu())
      
      '''
      ### find eigen-vectors
      with torch.enable_grad():
        xx = x.flatten().detach().clone()
        xx.requires_grad = True
        print(model.module.all_modules[2].weight.shape)
        print(model.module.all_modules[2])
        m = model.module.all_modules[2]
        y = m(xx.view(x.shape))
        print(xx.shape, y.shape)
        y = y.flatten()
        y[0].backward(retain_graph=True)
        
        grad_mat = np.load('assets/cifar10_eigen/grad_mat.npy')
        #grad_mat = np.zeros((y.numel(), xx.numel()))
        #print(grad_mat.shape)
        #for i in range(y.shape[0]):
        #  xx.grad.zero_()
        #  y[i].backward(retain_graph=True)
        #  grad_mat[i, :] = xx.grad.detach().cpu().numpy()
        #  if i % 10000 == 0:
        #    print(i) 
        #print(i)
        #np.save('assets/cifar10_eigen/grad_mat.npy', grad_mat)
        print(grad_mat.shape)
         
        grad_mat = grad_mat / np.linalg.norm(grad_mat, axis=1, keepdims=True) 
        print(np.linalg.norm(grad_mat[:, 0]))
        print(np.linalg.norm(grad_mat[0, :]))
        w = grad_mat.T.dot(grad_mat)
        print(w.shape)
        eigen_values, eigen_vectors = np.linalg.eig(w)
        print(eigen_vectors.shape)
        np.save('assets/cifar10_eigen/eigen_vectors.npy', eigen_vectors)
      assert False
      ### =====end======
      '''
      

      model_fn = mutils.get_model_fn(model, train=False) 
      
      
      # latent encoding
      '''
      N_rode = 1
      dt = 1./N_rode
      eps = 1e-3
      for i in range(N_rode):
        num_t = sde.T - i /N_rode * sde.T 
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999)
        x = x - pred * dt
      x = x.detach().clone()
      print('Latent:', x.min(), x.max())
      np.save('assets/cat_init.npy', x.cpu().numpy())
      assert False 
      '''

      ### Uniform
      dt = 1./sde.sample_N
      eps = 1e-3 # default: 1e-3
      #t_list = np.linspace(1e-3, sde.T, sde.N)
      for i in range(sde.sample_N):
        
        #t = torch.ones(shape[0], device=device) * t_list[i] #* i /sde.sample_N * (sde.T - 1e-3) + 1e-3
        num_t = i /sde.sample_N * (sde.T - eps) + eps
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999) ### Copy from models/utils.py 

        # convert to diffusion models
        sigma_t = sde.sigma_t(num_t)
        #print(sigma_t)
        pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.detach().clone())

        #x_pred = x.to(device) + (1. - t.item()) * pred_sigma

        x = x.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
        
        #print(num_t)
        #x[mask==1.0] = z0[mask==1.0] * (1.-num_t) + target[mask==1.0] * num_t

        #x_cllt.append(x[0:1].detach().clone().cpu())
        
        #import torchvision
        #torchvision.utils.save_image(torch.clamp(x, -1.0, 1.0), 'figs/2_real_sample_%d.png'%i, nrow=1, normalize=True) 
        #torchvision.utils.save_image(torch.clamp(x_pred, -1.0, 1.0), 'figs/2_target_sample_%d.png'%i, nrow=1, normalize=True) 
        
        '''
        if i <= 100:
          import seaborn as sns
          import matplotlib.pyplot as plt
          sns.kdeplot(x.view(-1).detach().clone().cpu().numpy())
        else:
          plt.savefig('figs/dist_%d.png'%i)
          assert False
        '''
        
      ''' 
      t_list = [0,1,2,5,10,15,20,25,50,75,100,300,500,800, 999]
      #dt_list = []
      for i in range(len(t_list)):
        
        import torchvision
        torchvision.utils.save_image(inverse_scaler(x_cllt[t_list[i]]).clamp_(0.0, 1.0), 'figs/real_%d.png'%t_list[i], nrow=1, normalize=True) 

        t = torch.ones(shape[0], device=device) * t_list[i]
        t = t /sde.sample_N * (sde.T - eps) + eps
        pred = model_fn(x_cllt[t_list[i]].to(device), t*999) ### Copy from models/utils.py 
        x_pred = x_cllt[t_list[i]].to(device) + (1. - t[0].item()) * pred
        torchvision.utils.save_image(inverse_scaler(x_pred).clamp_(0.0, 1.0), 'figs/pred_%d.png'%t_list[i], nrow=1, normalize=True)
      assert False 
      '''
      
      x = inverse_scaler(x)
      print(x.min(), x.max())
      
      import torchvision
      torchvision.utils.save_image(x.clamp_(0.0, 1.0), 'figs/cat_%d.png'%sde.sample_N, nrow=10, normalize=True) 
      assert False
      
      #print('Direction Variance:', get_direction_variance(x_cllt))
      #print('Relative OT loss', relative_OT_loss(torch.tensor(z0, device=device), x))
      #assert False
      
      ''' 
      import torchvision
      print(i+1)
      torchvision.utils.save_image(x, 'figs/sample_%d.png'%(i+1), nrow=10, normalize=True) 
      assert False
      '''
      
      ''' 
      x_cllt = torch.cat(x_cllt)
      print(x_cllt.shape)
      x_cllt = x_cllt.view(sde.sample_N+1, -1)
      x_cllt = x_cllt.cpu().numpy()
      print(x_cllt.shape)
      idx = np.linspace(0, 1, sde.sample_N+1)
      import matplotlib.pyplot as plt
      plt.figure(figsize=(2.5, 5.0))
      for i in range(20):
          plt.plot(idx, x_cllt[:, 0 + 150*i])
      
      plt.xticks([0.0, 1.0], fontsize=17)
      plt.yticks([-1.0, 0.0, 1.0], fontsize=17)
      plt.ylim(-1.5, 1.5)
      plt.xlabel('t', fontsize=20)
      plt.ylabel('Value', fontsize=20)

      plt.tight_layout()
      plt.savefig('figs/traj.png')
      assert False
      '''

      nfe = sde.sample_N
      return x, nfe
  
  def OT_loss(z0, generated_imgs):
    #loss = (z0 - generated_imgs).view(z0.shape[0], -1).norm(dim=1, keepdim=False) ### L2
    loss = (z0 - generated_imgs).view(z0.shape[0], -1).abs().sum(dim=1, keepdim=False)
    print(loss.shape)

    return loss.mean()

  def relative_OT_loss(z0, generated_imgs):
    x = z0.view(z0.shape[0], -1).cpu().numpy()
    y = generated_imgs.view(generated_imgs.shape[0], -1).cpu().numpy()
    #print(x.shape, y.shape)
    cost=np.sum((x[:,None,:] - y[None,:,:])**2, -1)
    #print(cost.shape, x.shape, y.shape)
    print(cost)
    #x0 = torch.tensor(x[1:2]).repeat(z0.shape[0], 1)
    #print(cost[1])
    #print(x0.shape)
    #print((x0-torch.tensor(y)).pow(2).sum(dim=-1))
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    print(row_ind)
    print(col_ind)
    #assert False
    ot_loss = np.mean(np.sum((x[row_ind,:]-y[col_ind,:])**2,-1))
    loss = np.mean(np.sum((x-y)**2,-1))
    print('loss:', loss, 'OT loss:', ot_loss)
    assert False

    return loss.mean() / ot_loss.mean()

  def get_direction_variance(traj):
    traj = torch.stack(traj)
    print(traj.shape)
    traj = torch.nn.functional.normalize(traj, dim=-1)
    traj = traj.std(dim=0)
    traj = traj.mean(dim=0)
    print(traj.shape)

    return traj.mean()
  
  def local_smoothness(d_input, d_output):
    print(d_input.shape, d_output.shape)
    est = d_output / d_input
    est = est.view(est.shape[0], -1).norm(dim=1)
    print(est.shape)

    return est.mean()

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      rtol=atol=sde.ode_tol
      method='RK45'
      eps=1e-3

      # Initial sample
      if z is None:
        z0 = sde.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
       
        print(x.shape, x.min(), x.max())
      else:
        x = z
      
      model_fn = mutils.get_model_fn(model, train=False)

      
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)

     
        return to_flattened_numpy(drift)

      
      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)


      x = inverse_scaler(x)
      print(x.min(), x.max(), x.shape, nfe)

      #import torchvision
      #torchvision.utils.save_image(x.clamp_(0.0, 1.0), 'figs/samples.png', nrow=10, normalize=True) 
      #assert False

      return x, nfe
  
  print('Use ODE?', sde.use_ode_sampler)
  if sde.use_ode_sampler:
      return ode_sampler
  else:
      return mixup_sampler
