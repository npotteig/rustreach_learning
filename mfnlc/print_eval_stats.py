import pandas as pd

from mfnlc.config import get_path

ROBOT_NAME = "Bicycle"
ALGO = "rrt_lyapunov"

if __name__ == "__main__":
    res_path = get_path(robot_name=ROBOT_NAME, algo=ALGO, task="evaluation") + "/line_exp/line_eval_output.csv"
    
    df = pd.read_csv(res_path)
    
    ttg_df = df[(df["total_step"] < 200) & (df["collision"] == 0.0)]
    print("Average time to reach goal (TTG) without safety violation and timeouts:", ttg_df["TTG"].mean())
    print("Timeouts:", len(df[df["total_step"] == 200]["total_step"]))
    print("Collisions:", len(df[df["collision"] == 1.0]["collision"]))
    print("Average subgoal computation time:", df["Avg Subgoal Compute Time"].mean())
    print("Max subgoal computation time:", df["Max Subgoal Compute Time"].max())
    print("Total deadline violations:", df["Deadline Violations"].sum())