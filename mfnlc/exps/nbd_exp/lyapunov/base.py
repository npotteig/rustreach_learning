import random
from typing import Dict, List
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from mfnlc.config import get_path
from mfnlc.envs import get_env
from mfnlc.envs.base import ObstacleMaskWrapper
from mfnlc.envs.difficulty import choose_level
from mfnlc.evaluation.model import load_model
from mfnlc.evaluation.simulation import simu
from mfnlc.learn.lyapunov_td3 import LyapunovTD3
from mfnlc.monitor.monitor import Monitor, LyapunovValueTable
from mfnlc.plan.common.path import Path

ALGO = "rrt_lyapunov"
OBSTACLE_SPEED = 0.5
DT = 0.1
perp_slope_dx = 0
perp_slope_dy = 0

def evaluate(env_name,
             n_rollout: int = 1,
             n_steps: int = 1000,
             arrive_radius: float = 0.1,
             monitor_max_step_size: float = 0.2,
             monitor_search_step_size: float = 0.01,
             render: bool = False,
             render_config: Dict = {},  # noqa
             seed: int = None,
             path_dataset: List = None,
             obstacle_dataset: np.ndarray = None,
             dynamic_obstacles: bool = False,
             ):
    model: LyapunovTD3 = load_model(env_name, algo=ALGO)
    env = ObstacleMaskWrapper(get_env(env_name))
    
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    choose_level(env, 1)
    
    lv_table = LyapunovValueTable.load(get_path(env.robot_name, ALGO, "lv_table"))
    monitor = Monitor(lv_table, max_step_size=monitor_max_step_size, search_step_size=monitor_search_step_size)
    
    i = 0
    running_data = {
        "total_step": [],
        "TTG": [],
        "goal_met": [],
        "collision": [],
        "reward_sum": [],
        "Avg Subgoal Compute Time": [],
        "Max Subgoal Compute Time": [],
        "Deadline Violations": [],
    }
    
    if dynamic_obstacles:
        update_obstacles = update_obstacles_dynamic
        set_obs_pos = set_obs_pos_dynamic
    else:
        update_obstacles = update_obstacles_static
        set_obs_pos = set_obs_pos_static
    
    for pth in tqdm(path_dataset):
        env.reset()
        path: Path = Path(np.array(pth)[:, :2].tolist())
        
        env.unwrapped.set_goal(np.array(pth[-1][:2]))
        env.unwrapped.set_robot_pos(np.array(pth[0][:2]))
        obstacle_set = obstacle_dataset[obstacle_dataset[:, 0] == i][:, 1:]
        if obstacle_set.size == 0:
            obstacle_set = np.array([[1e5, 1e5]])
        env.unwrapped.set_obstacle_centers(obstacle_set)
        
        res = simu(env=env,
                   model=model,
                   update_obstacles=update_obstacles,
                   n_steps=n_steps,
                   path=path,
                   arrive_radius=arrive_radius,
                   monitor=monitor,
                   render=render,
                   render_config=render_config,
                   set_obs_pos=set_obs_pos)
        for k in res:
            running_data[k].append(res[k])
        i += 1
        
    stat = pd.DataFrame(running_data)

    res_dir = get_path(robot_name=env.robot_name, algo=ALGO, task="evaluation") + "/nbd_exp"
    os.makedirs(res_dir, exist_ok=True)
    obstacle_type = "dynamic" if dynamic_obstacles else "static"
    stat.to_csv(res_dir + f"/mfnlc_{obstacle_type}_nbd_output.csv")
    print("results are saved to:", res_dir + f"/mfnlc_{obstacle_type}_nbd_output.csv")

    env.close()

def build_lyapunov_table(env_name: str,
                         obs_lb: np.ndarray,
                         obs_ub: np.ndarray,
                         n_levels: int = 10,
                         pgd_max_iter: int = 100,
                         pgd_lr: float = 1e-3,
                         n_range_est_sample: int = 10,
                         n_radius_est_sample: int = 10,
                         bound_cnst: float = 100):
    model: LyapunovTD3 = load_model(env_name, algo=ALGO)
    lv_table = LyapunovValueTable(model.tclf,
                                  obs_lb,
                                  obs_ub,
                                  n_levels=n_levels,
                                  pgd_max_iter=pgd_max_iter,
                                  pgd_lr=pgd_lr,
                                  n_range_est_sample=n_range_est_sample,
                                  n_radius_est_sample=n_radius_est_sample,
                                  bound_cnst=bound_cnst)
    lv_table.build()
    print(lv_table.lyapunov_values)
    print(lv_table.lyapunov_radius)
    robot_name = env_name.split("-")[0]
    lv_table.save(get_path(robot_name, algo=ALGO, task="lv_table"))

def update_obstacles_static(env):
    pass

def update_obstacles_dynamic(env):
    global perp_slope_dx, perp_slope_dy
    obstacles = env.unwrapped.hazards_pos
    offset = OBSTACLE_SPEED * DT

    obstacles[0, 0] -= perp_slope_dx*offset
    obstacles[0, 1] -= perp_slope_dy*offset
    env.unwrapped.set_obstacle_centers(obstacles)
    
def set_obs_pos_static(env, prev_waypoint, next_waypoint):
    pass

def set_obs_pos_dynamic(env, prev_waypoint, next_waypoint):
    global perp_slope_dx, perp_slope_dy
    mid_x = (prev_waypoint[0] + next_waypoint[0]) / 2
    mid_y = (prev_waypoint[1] + next_waypoint[1]) / 2
    
    new_x = 0
    new_y = 0
    offset = 2
    if prev_waypoint[0] == next_waypoint[0]:
        new_x = mid_x + offset
        new_y = mid_y
        perp_slope_dx = 1
        perp_slope_dy = 0
    elif prev_waypoint[1] == next_waypoint[1]:
        new_x = mid_x
        new_y = mid_y + offset
        perp_slope_dx = 0
        perp_slope_dy = 1
    else:
        dx = next_waypoint[0] - prev_waypoint[0]
        dy = next_waypoint[1] - prev_waypoint[1]
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length
        perp_slope_dx = nx
        perp_slope_dy = ny
        
        new_x = mid_x + offset * nx
        new_y = mid_y + offset * ny
    
    obstacles = env.unwrapped.hazards_pos
    obstacles[0, 0] = new_x
    obstacles[0, 1] = new_y
    env.unwrapped.set_obstacle_centers(obstacles)