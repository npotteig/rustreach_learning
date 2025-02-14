import pandas as pd

from mfnlc.config import get_path

ROBOT_NAME = "Quadcopter"
OBSTACLE_TYPE = "dynamic"
ALGO = "rrt_lyapunov"
CORR_EXP = False
TOTAL_STEPS = 200 if CORR_EXP else 1000


if __name__ == "__main__":
    if CORR_EXP:
        pth = f"/corr_exp/mfnlc_{OBSTACLE_TYPE}_corr_output.csv"
    else:
        pth = f"/nbd_exp/mfnlc_{OBSTACLE_TYPE}_nbd_output.csv"
    res_path = get_path(robot_name=ROBOT_NAME, algo=ALGO, task="evaluation") + pth
    
    df = pd.read_csv(res_path)
    
    ttg_df = df[(df["total_step"] < TOTAL_STEPS) & (df["collision"] == 0.0)]
    print("Average time to reach goal (TTG) without safety violation and timeouts:", ttg_df["TTG"].mean())
    print("Timeouts:", len(df[df["total_step"] == TOTAL_STEPS]["total_step"]))
    print("Collisions:", len(df[df["collision"] == 1.0]["collision"]))
    print("Average subgoal computation time:", df["Avg Subgoal Compute Time"].mean())
    print("Max subgoal computation time:", df["Max Subgoal Compute Time"].max())
    print("Total deadline violations:", df["Deadline Violations"].sum())