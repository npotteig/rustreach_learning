from typing import Dict, Callable
import time

import numpy as np

from mfnlc.config import env_config
from mfnlc.envs import get_env
from mfnlc.evaluation.model import load_model
from mfnlc.monitor.monitor import Monitor
from mfnlc.plan.common.path import Path

def inspect_training_simu(env_name: str,
                          algo: str,
                          n_rollout: int,
                          render: bool = False):
    env = get_env(env_name)
    robot_name = env_name.split("-")[0]

    model = load_model(env_name, algo)

    simu_data = {
        "rewards": [],
        "obs": [],
        "actions": [],
        "infos": []
    }

    for ep in range(n_rollout):
        reward_list = []
        obs_list = []
        action_list = []
        info_list = []

        obs = env.reset()
        obs_list.append(obs)
        for i in range(env_config[robot_name]["max_step"]):
            action = model.predict(obs)[0]
            obs, reward, done, info = env.step(action)

            action_list.append(action)
            obs_list.append(obs)
            reward_list.append(reward)
            info_list.append(info)

            if render:
                env.render()

            if done:
                break

        simu_data["obs"].append(obs_list)
        simu_data["actions"].append(action_list)
        simu_data["rewards"].append(reward_list)
        simu_data["infos"].append(info_list)

    return simu_data


def simu(env,
         model,
         update_obstacles: Callable,
         n_steps: int,
         path: Path = None,
         arrive_radius: float = 0.0,
         monitor: Monitor = None,
         render: bool = False,
         render_config: Dict = {},  # noqa
         set_obs_pos: Callable = None,
         ):
    obs = env.get_obs()

    if monitor is not None:
        monitor.reset()

    subgoal_index = 0
    if path is not None:
        env.set_subgoal(path[subgoal_index])
    if set_obs_pos is not None and np.linalg.norm(env.robot_pos - path[subgoal_index]) > 3.0:
        set_obs_pos(env, path[subgoal_index-1], path[subgoal_index])

    env.set_render_config(render_config)
    if render:
        env.render()

    total_step = 0
    goal_met = False
    collision = False
    reward_sum = 0.0
    deadline_violations = 0
    subgoal_compute_times = []

    for i in range(n_steps):
        action = model.predict(obs)[0]
        obs, reward, done, info = env.step(action)
        total_step += 1
        reward_sum += reward

        if path is not None:
            if np.linalg.norm(env.robot_pos - path[subgoal_index]) < arrive_radius:
                subgoal_index += 1
                if subgoal_index == len(path):
                    subgoal_index -= 1
                else:
                    if set_obs_pos is not None and np.linalg.norm(path[subgoal_index-1] - path[subgoal_index]) > 4.0:
                        set_obs_pos(env, path[subgoal_index-1], path[subgoal_index])
                # subgoal_index = min(len(path) - 1, subgoal_index)
                subgoal = path[subgoal_index]
                env.set_subgoal(subgoal, store=True)
            else:
                subgoal = path[subgoal_index]
            if monitor is not None:
                start_time_us = int(time.time() * 1e6)
                subgoal, lyapunov_r = monitor.select_subgoal(env, subgoal)
                duration = int(time.time() * 1e6) - start_time_us
                if duration >= 100_000:
                    deadline_violations += 1
                subgoal_compute_times.append(duration)
            env.set_subgoal(subgoal, store=False)
            env.set_roa(subgoal, lyapunov_r)  # noqa
            update_obstacles(env)

        if render:
            if path is not None and monitor is not None:
                env.render()
            else:
                env.render()

        if done:
            goal_met = info.get("goal_met", False)
            collision = info.get("collision", False)
            break

    return {"total_step": total_step,
            "TTG": env.unwrapped.step_size * total_step,
            "collision": float(collision),
            "goal_met": goal_met,
            "reward_sum": reward_sum,
            "Avg Subgoal Compute Time": np.mean(subgoal_compute_times) if subgoal_compute_times else 0,
            "Max Subgoal Compute Time": np.max(subgoal_compute_times) if subgoal_compute_times else 0,
            "Deadline Violations": deadline_violations}
    
        
