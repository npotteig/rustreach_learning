import numpy as np
import pandas as pd

from mfnlc.exps.line_exp.lyapunov.base import evaluate, build_lyapunov_table

ENV_NAME = "Bicycle-eval"

def lyapunov_eval():
    line_dataset_path = "rustreach_exp_data/line_exp/bicycle/line_dataset.csv"
    print(f"{ENV_NAME} - Lyapunov-TD3")
    evaluate(ENV_NAME,
                n_rollout=1,
                render_config={
                    "traj_sample_freq": 10,
                },
                monitor_max_step_size=0.4,
                monitor_search_step_size=5e-3,
                n_steps=200,
                arrive_radius=0.2,
                render=False,
                seed=0,
                line_dataset=pd.read_csv(line_dataset_path))

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