"""Self-Flow Sampling Utilities (adapted from black-forest-labs/Self-Flow, Apache-2.0,
extended with ODE Euler/Heun + Appendix A.2/A.4 timeshift sampling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import Literal, Optional

import numpy as np
import torch

from self_flow.schedules import timeshift

Choice_PathType = Literal["Linear", "GVP", "VP"]
Choice_Prediction = Literal["velocity", "score", "noise"]
Choice_LossWeight = Optional[Literal["velocity", "likelihood"]]
Choice_SamplingODE = Literal["euler", "heun2"]
Choice_SamplingSDE = Literal["Euler", "Heun"]
Choice_Diffusion = Literal[
    "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"
]
Choice_LastStep = Optional[Literal["Mean", "Tweedie", "Euler"]]


@dataclass
class TransportConfig:
    path_type: Choice_PathType = "Linear"
    prediction: Choice_Prediction = "velocity"
    loss_weight: Choice_LossWeight = None
    sample_eps: Optional[float] = None
    train_eps: Optional[float] = None


@dataclass
class ODEConfig:
    sampling_method: Choice_SamplingODE = "heun2"
    sample_shift: float = 1.0


@dataclass
class SDEConfig:
    sampling_method: Choice_SamplingSDE = "Euler"
    diffusion_form: Choice_Diffusion = "sigma"
    diffusion_norm: float = 1.0
    last_step: Choice_LastStep = "Mean"
    last_step_size: float = 0.04


@dataclass
class Config:
    transport: TransportConfig = field(default_factory=TransportConfig)
    ode: ODEConfig = field(default_factory=ODEConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    num_steps: int = 64
    cfg_scale: float = 1.0


def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


class ICPlan:
    """Linear Coupling Plan (Interpolant Conditional Plan)."""

    def __init__(self, sigma: float = 0.0) -> None:
        self.sigma = sigma

    def compute_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, int]:
        return t, 1

    def compute_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, int]:
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        return 1 / t

    def compute_drift(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(
        self, x: torch.Tensor, t: torch.Tensor, form: str = "constant", norm: float = 1.0
    ) -> torch.Tensor | float:
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }
        if form not in choices:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        return choices[form]

    def get_score_from_velocity(
        self, velocity: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * velocity - x) / var


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps) -> None:
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = {PathType.LINEAR: ICPlan}[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form: str = "SBDM",
        sde: bool = False,
        reverse: bool = False,
        eval: bool = False,
        last_step_size: float = 0.0,
    ) -> tuple[float, float]:
        t0, t1 = 0.0, 1.0
        eps = train_eps if not eval else sample_eps
        if self.model_type != ModelType.VELOCITY or sde:
            t0 = (
                eps
                if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY
                else 0.0
            )
            t1 = 1.0 - eps if (not sde or last_step_size == 0.0) else 1.0 - last_step_size
        if reverse:
            t0, t1 = 1.0 - t0, 1.0 - t1
        return t0, t1

    def get_drift_from_model_output(self):
        return lambda x, t, model_output: model_output

    def get_score_from_model_output(self):
        return lambda x, t, model_output: self.path_sampler.get_score_from_velocity(
            model_output, x, t
        )


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
) -> Transport:
    return Transport(
        model_type=ModelType.VELOCITY,
        path_type={"Linear": PathType.LINEAR}[path_type],
        loss_type=WeightType.NONE,
        train_eps=0.0,
        sample_eps=0.0,
    )


class sde:
    def __init__(self, drift, diffusion, *, t0: float, t1: float, num_steps: int, sampler_type: str):
        assert t0 < t1, "SDE sampler has to be in forward time"
        self.num_timesteps = num_steps
        self.t = torch.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def _euler_maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn_like(x)
        t_vec = torch.ones(x.size(0), device=x.device, dtype=x.dtype) * t
        dw = w_cur * torch.sqrt(self.dt.to(x.device))
        drift = self.drift(x, t_vec, model, **model_kwargs)
        diffusion = self.diffusion(x, t_vec)
        mean_x = x + drift * self.dt.to(x.device)
        x = mean_x + torch.sqrt(2.0 * diffusion) * dw
        return x, mean_x

    def _heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = torch.randn_like(x)
        dt = self.dt.to(x.device)
        dw = w_cur * torch.sqrt(dt)
        t_cur = torch.ones(x.size(0), device=x.device, dtype=x.dtype) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2.0 * diffusion) * dw
        k1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + dt * k1
        k2 = self.drift(xp, t_cur + dt, model, **model_kwargs)
        return xhat + 0.5 * dt * (k1 + k2), xhat

    def sample(self, init: torch.Tensor, model, **model_kwargs) -> list[torch.Tensor]:
        x = init
        mean_x = init
        samples = []
        step_fn = {"Euler": self._euler_maruyama_step, "Heun": self._heun_step}[self.sampler_type]
        for ti in self.t[:-1]:
            with torch.no_grad():
                x, mean_x = step_fn(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


class FixedSampler:
    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.drift = self.transport.get_drift_from_model_output()
        self.score = self.transport.get_score_from_model_output()

    def _get_sde_diffusion_and_drift(self, *, diffusion_form: str = "SBDM", diffusion_norm: float = 1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

        def sde_drift(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)
            return self.drift(x, t, model_output) + diffusion_fn(x, t) * self.score(
                x, t, model_output
            )

        return sde_drift, diffusion_fn

    def sample_sde(
        self,
        *,
        sampling_method: str = "Euler",
        diffusion_form: str = "sigma",
        diffusion_norm: float = 1.0,
        last_step: str | None = "Mean",
        last_step_size: float = 0.04,
        num_steps: int = 250,
    ):
        if last_step is None:
            last_step_size = 0.0
        sde_drift, sde_diffusion = self._get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )
        _sde = sde(
            sde_drift, sde_diffusion, t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method
        )

        def _sample(init: torch.Tensor, model, **model_kwargs) -> list[torch.Tensor]:
            xs = _sde.sample(init, model, **model_kwargs)
            ts = torch.ones(init.size(0), device=init.device, dtype=init.dtype) * t1
            if last_step == "Mean":
                x = xs[-1] + sde_drift(xs[-1], ts, model, **model_kwargs) * last_step_size
            elif last_step == "Euler":
                x = xs[-1] + self.drift(xs[-1], ts, model(xs[-1], ts, **model_kwargs)) * last_step_size
            else:
                x = xs[-1]
            xs.append(x)
            return xs

        return _sample


def vanilla_guidance(x: torch.Tensor, cfg_val: float | torch.Tensor) -> torch.Tensor:
    x_u, x_c = x.chunk(2)
    if isinstance(cfg_val, torch.Tensor):
        while cfg_val.dim() < x_u.dim():
            cfg_val = cfg_val.unsqueeze(-1)
    return x_u + cfg_val * (x_c - x_u)


@torch.no_grad()
def sample_ode(
    init: torch.Tensor,
    model_eval_tau,
    *,
    num_steps: int = 50,
    sample_shift: float = 1.0,
    method: str = "euler",
) -> torch.Tensor:
    """Rectified-flow ODE solver from tau=1 (noise x_1) -> tau=0 (clean x_0) with timeshift (Eq. 8)."""
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=init.device, dtype=torch.float32)
    ts = timeshift(ts, alpha=sample_shift)
    x = init
    b = x.shape[0]
    for i in range(num_steps):
        tau_curr = ts[i].expand(b)
        tau_next = ts[i + 1].expand(b)
        dt = (tau_next - tau_curr).view(b, *([1] * (x.ndim - 1))).to(x.dtype)
        v1 = model_eval_tau(x, tau_curr)
        if method == "heun2" and i < num_steps - 1:
            x_euler = x + dt * v1
            v2 = model_eval_tau(x_euler, tau_next)
            x = x + 0.5 * dt * (v1 + v2)
        else:
            x = x + dt * v1
    return x


def denoise_loop(
    *,
    model,
    num_steps: int,
    cfg_scale: float | None = None,
    guidance_low: float = 0.0,
    guidance_high: float = 1.0,
    mode: str = "SDE",
    sampling_method: str = "euler",
    sample_shift: float = 1.0,
    reverse: bool = True,
    **model_kwargs,
) -> torch.Tensor:
    """Main Self-Flow denoising loop supporting both SDE (250-step ImageNet) and ODE (timeshifted)."""
    x_init = model_kwargs.pop("x")

    def eval_paper_velocity(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if cfg_scale is not None and cfg_scale > 1.0:
            tau_scalar = float(tau.reshape(-1)[0].item())
            apply_cfg = guidance_low <= tau_scalar <= guidance_high
        else:
            apply_cfg = False

        if apply_cfg:
            bs = x.shape[0]
            assert bs % 2 == 0
            x_in = torch.cat((x[bs // 2 :], x[bs // 2 :]), dim=0)
        else:
            x_in = x

        pred = model(x_in, timesteps=tau, **model_kwargs).to(torch.float32)
        if apply_cfg:
            pred = vanilla_guidance(pred, cfg_val=cfg_scale)
            pred = torch.cat((pred, pred), dim=0)
        return pred

    if mode == "ODE":
        return sample_ode(
            x_init,
            eval_paper_velocity,
            num_steps=num_steps,
            sample_shift=sample_shift,
            method=sampling_method,
        )

    if mode != "SDE":
        raise ValueError(f"Unsupported mode {mode!r}; expected 'SDE' or 'ODE'.")

    args = Config(num_steps=num_steps)
    transport = create_transport(
        args.transport.path_type,
        args.transport.prediction,
        args.transport.loss_weight,
        args.transport.train_eps,
        args.transport.sample_eps,
    )
    sampler = FixedSampler(transport)
    sample_fn = sampler.sample_sde(
        sampling_method=args.sde.sampling_method,
        diffusion_form=args.sde.diffusion_form,
        diffusion_norm=args.sde.diffusion_norm,
        last_step=args.sde.last_step,
        last_step_size=args.sde.last_step_size,
        num_steps=args.num_steps,
    )

    def sde_model_fn(x: torch.Tensor, t_sde: torch.Tensor, **_) -> torch.Tensor:
        tau = 1.0 - t_sde if reverse else t_sde
        pred = eval_paper_velocity(x, tau)
        return -pred if reverse else pred

    return sample_fn(x_init, sde_model_fn)[-1]
