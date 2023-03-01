from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_activation: torch.nn.functional = F.relu,
        init_w: float = 3e-3,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation

        # Set fully connected layers
        self.fc_layers = nn.ModuleList()
        self.hidden_layers = [hidden_dim] * num_hidden_layers  # 此时的hidden_layers只是一个int列表
        in_layer = input_dim

        for i, hidden_layer in enumerate(self.hidden_layers):
            fc_layer = nn.Linear(in_layer, hidden_layer)
            in_layer = hidden_layer  # 上一层的输出维度就是下一层的输入维度
            self.__setattr__("fc_layer{}".format(i), fc_layer)  # 创建一个名为fc_layeri的属性, 其值为nn.Linear
            self.fc_layers.append(fc_layer)
        # 此时的self.fc_layers已装填好nn.Linear

        # 定义输出层
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        is_deterministic: bool = False,
        init_w: float = 1e-3,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )

        self.is_deterministic = is_deterministic
        self.last_fc_log_std = nn.Linear(hidden_dim, output_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        # 专门定义一个输出层, 用于输出log_std
        # 相等于mean和log_std共享隐藏层, 分别用两个输出层输出

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))

        mean = self.last_fc_layer(x)
        log_std = self.last_fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        # torch.clamp对log_std进行截断, 使得log_std的值在LOG_SIG_MIN和LOG_SIG_MAX之间
        std = torch.exp(log_std)

        if self.is_deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)  # torch.distributions.Normal
            action = normal.rsample()  # 从正态中采样, 输入为shape, 不写shape则采样1次

            log_prob = normal.log_prob(action)  # 计算采样到的action的对数概率(即采取这个action的概率的对数)
            log_prob -= 2 * (np.log(2) - action - F.softplus(-2 * action))  # softplus(x) = log(1 + exp(x))
            log_prob = log_prob.sum(-1, keepdim=True)

        action = torch.tanh(action)
        return action, log_prob


class FlattenMLP(MLP):
    def forward(self, *x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=-1)
        return super().forward(x)


class MLPEncoder(FlattenMLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,  # z的维度
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.device = device

        self.z_mean = None
        self.z_var = None  # variance
        self.task_z = None
        self.context = None
        self.clear_z()

    def clear_z(self, num_tasks: int = 1) -> None:
        # 设置z的先验分布, N(0, 1)
        self.z_mean = torch.zeros(num_tasks, self.latent_dim).to(self.device)
        self.z_var = torch.ones(num_tasks, self.latent_dim).to(self.device)

        # 从先验分布中采样z
        self.sample_z()

        # 重置context
        self.context = None

    def sample_z(self) -> None:
        # Sample z ~ r(z) or z ~ q(z|c)
        dists = []
        for mean, var in zip(torch.unbind(self.z_mean), torch.unbind(self.z_var)):
            dist = torch.distributions.Normal(mean, torch.sqrt(var))
            dists.append(dist)
        sampled_z = [dist.rsample() for dist in dists]
        self.task_z = torch.stack(sampled_z).to(self.device)

    @classmethod
    def product_of_gaussians(
        cls,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute mean, stddev of product of gaussians (POG)
        var = torch.clamp(var, min=1e-7)
        pog_var = 1.0 / torch.sum(torch.reciprocal(var), dim=0)  # 倒数之和的倒数
        #
        pog_mean = pog_var * torch.sum(mean / var, dim=0)
        return pog_mean, pog_var

    def infer_posterior(self, context: torch.Tensor) -> None:
        # 从context中推断出z的后验分布
        params = self.forward(context)
        params = params.view(context.size(0), -1, self.output_dim).to(self.device)

        # With probabilistic z, predict mean and variance of q(z | c)
        z_mean = torch.unbind(params[..., : self.latent_dim])
        z_var = torch.unbind(F.softplus(params[..., self.latent_dim :]))
        z_params = [self.product_of_gaussians(mu, var) for mu, var in zip(z_mean, z_var)]

        self.z_mean = torch.stack([z_param[0] for z_param in z_params]).to(self.device)
        self.z_var = torch.stack([z_param[1] for z_param in z_params]).to(self.device)
        self.sample_z()

    def compute_kl_div(self) -> torch.Tensor:
        # Compute KL( q(z|c) || r(z) )
        prior = torch.distributions.Normal(
            torch.zeros(self.latent_dim).to(self.device),
            torch.ones(self.latent_dim).to(self.device),
        )

        posteriors = []
        for mean, var in zip(torch.unbind(self.z_mean), torch.unbind(self.z_var)):
            dist = torch.distributions.Normal(mean, torch.sqrt(var))
            posteriors.append(dist)

        kl_div = [torch.distributions.kl.kl_divergence(posterior, prior) for posterior in posteriors]
        kl_div = torch.stack(kl_div).sum().to(self.device)
        return kl_div



