import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from mfnlc.evaluation.simulation import inspect_training_simu
from mfnlc.exps.train.no_obstacle.lyapunov_td3.base import train, evaluate_lyapunov_of

def colearn():
    action_noise = OrnsteinUhlenbeckActionNoise(np.array([0, 0]), np.array([0.2, 0.2]))
    train(env_name="Bicycle-no-obst",
          lf_structure=[4, 64, 64, 1],
          lqf_structure=[4 + 2, 64, 64, 1],
          tclf_ub=15,
          tclf_q_sigma=0,
        #   lqf_loss_cnst=0.2,
          total_timesteps=100_000,
          action_noise=action_noise,
        #   batch_size=1024,
          policy_kwargs={"net_arch": [32, 32]})

def evaluate_controller():
    inspect_training_simu(env_name="Bicycle-no-obst",
                          algo="lyapunov_td3",
                          n_rollout=5,
                          render=True)


def evaluate_lyapunov():
    evaluate_lyapunov_of("Bicycle")


if __name__ == "__main__":
    # for i in range(5):
    colearn()
    # evaluate_controller()
    # evaluate_lyapunov()