import numpy as np
import pandas as pd

from mfnlc.exps.nbd_exp.lyapunov.base import evaluate, build_lyapunov_table

ENV_NAME = "Bicycle-eval"

def lyapunov_eval():
    path_dataset_path = "rustreach_exp_data/nbd_exp/astar_rustreach_paths.csv"
    path_data = np.loadtxt(path_dataset_path, delimiter=",")
    obstacle_dataset_path = "rustreach_exp_data/nbd_exp/rr_nbd_obstacles_near_path.csv"
    obstacle_data = np.loadtxt(obstacle_dataset_path, delimiter=",")
    print(f"{ENV_NAME} - Lyapunov-TD3")
    evaluate(ENV_NAME,
                n_rollout=1,
                render_config={
                    "traj_sample_freq": 10,
                },
                monitor_max_step_size=0.4,
                monitor_search_step_size=5e-3,
                n_steps=1000,
                arrive_radius=1.0,
                render=False,
                seed=0,
                path_dataset=[path_data[path_data[:, 0] == i][:, 1:].tolist() for i in range(0, 1000)],
                obstacle_dataset=obstacle_data,
                dynamic_obstacles=True,)

def build_lv_table():
    lb = np.array([-1, -1, -5, -5])
    ub = np.array([1, 1, 5, 5])
    build_lyapunov_table(ENV_NAME,
                         lb, ub,
                         pgd_max_iter=500,
                         n_radius_est_sample=20)

if __name__ == '__main__':
    # build_lv_table()
    lyapunov_eval()