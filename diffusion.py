import torch   
import torch.nn as nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_hat = self.alpha_hat[t] ** 0.5
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]) ** 0.5

        return sqrt_alpha_hat[:, None, None, None] * x0 + \
               sqrt_one_minus_alpha_hat[:, None, None, None] * noise
    
    def p_sample(self, model, x, t, cond):
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_hat_t = self.alpha_hat[t]

        eps_theta = model(x, t.float(), cond)

        coef1 = 1. / (alpha_t.sqrt())
        coef2 = beta_t / (1 - alpha_hat_t).sqrt()

        mean = coef1[:, None, None, None] * \
               (x - coef2[:, None, None, None] * eps_theta)

        if t[0] == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            return mean + (beta_t.sqrt())[:, None, None, None] * noise

    def sample(self, model, cond, shape):
        x = torch.randn(shape).to(cond.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t]*shape[0]).to(cond.device)
            x = self.p_sample(model, x, t_tensor, cond)
        return x