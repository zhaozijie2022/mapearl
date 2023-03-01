import os
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from mpe.lib4occupy import *

from meta_rl.envs import ENVS
from pearl.meta_learner import MetaLearner
from pearl.SacAgent import SacAgent


num_tasks = 100


if __name__ == "__main__":
    # 初始化实验参数
    args_exp = parse_args_exp(
        exp_name="mt-occupy",
        num_train_tasks=100,
        num_test_tasks=30,
        latent_dim=5
    )
    args_pearl = parse_args_pearl(
        num_epochs=1000,
        num_sample_tasks=5,
        num_init_samples=2000,
        num_prior_samples=400,
        num_posterior_samples=600,
        num_meta_grads=1500,
        meta_batch_size=16,
    )
    args_sac = parse_args_sac(
        gamma=0.99,
        kl_lambda=0.1,
        batch_size=256,
        qf_lr=3e-4,
        policy_lr=3e-4,
        vf_lr=3e-4,
    )

    # 生成多任务环境, 任务采样
    envs = make_mtenv("mt-occupy", num_tasks=args_exp.num_train_tasks + args_exp.num_test_tasks)
    # 返回一个列表, 列表中的每个元素是一个MultiTaskEnv, 
    # 继承自mpe.environment.MultiAgentEnv, 仅新增task_index属性, 用于标记任务

    obs_dim = envs[0].observation_space[0].shape[0]
    # obs_dim = env.n * env.world.dim_p (all agent pos) + 2 (self vel), landmark不可见
    act_dim = envs[0].action_space[0].n
    hid_dim = 32

    device = torch.device(torch.device("cuda")
                          if torch.cuda.is_available()
                          else torch.device("cpu"))

    agent = SacAgent(
        observ_dim=obs_dim,
        action_dim=act_dim,
        latent_dim=5,  # 上下文变量的维度
        hidden_dim=hid_dim,
        encoder_input_dim=obs_dim + act_dim + 1,
        encoder_output_dim=5 * 2,
        device=device,
        args_sac=args_sac,
    )

    meta_learner = MetaLearner(
        env=envs,
        env_name=experiment_config["env_name"],
        agent=agent,
        observ_dim=observ_dim,
        action_dim=action_dim,
        train_tasks=tasks[: env_target_config["train_tasks"]],
        test_tasks=tasks[-env_target_config["test_tasks"] :],
        save_exp_name=experiment_config["save_exp_name"],
        save_file_name=experiment_config["save_file_name"],
        load_exp_name=experiment_config["load_exp_name"],
        load_file_name=experiment_config["load_file_name"],
        load_ckpt_num=experiment_config["load_ckpt_num"],
        device=device,
        **env_target_config["pearl_params"],
    )

    # Run PEARL training
    meta_learner.meta_train()
