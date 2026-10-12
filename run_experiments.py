
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from pathlib import Path



import tools
from env import GridworldEnv
from value_iteration import value_iteration
from policy_iteration import policy_iteration
from montecarlo_adv import montecarlo_adv
from td_learning import evaluate_policy, show_policy
from SARSA import train_sarsa
from Q_learning import train_q_learning

GAMMA = 0.99
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 100
MC_EPISODES = 5000
MC_EPSILON = 0.3            
OUT = "results"


def moving_average(x, w=WINDOW):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    os.makedirs(OUT, exist_ok=True)

    params_path = Path("best_sarsa_params.json")

    if not params_path.exists():
        raise FileNotFoundError(
            "Can't find best_sarsa_params.json. "
            "First execute optimizacion.py."
        )

    with params_path.open("r", encoding="utf-8") as f:
        best_params = json.load(f)
    rows = []

    for tag, slippery in (("deterministic", False), ("stochastic", True)):
        env = GridworldEnv(is_slippery=slippery)
        print(f"\n{'=' * 64}\nENV: {tag}  (is_slippery={slippery})\n{'=' * 64}")

        
        V_star, Q_star, pi_star, _ = value_iteration(env.P, gamma=GAMMA, epsilon=1e-8)
        _, _, pi_pi = policy_iteration(env.P, gamma=GAMMA)
        print(f"PI policy == VI policy: {np.array_equal(pi_pi, pi_star)}")

        sr, ret, steps = evaluate_policy(Q_star, env)
        print(f"Optimal (VI):  success={sr:.3f}  return={ret:.4f}  steps={steps:.2f}")
        show_policy(Q_star, f"π* ({tag}, Value Iteration)")
        tools.plot_value_function(V_star, env, title=f"V* — {tag} (VI)")
        tools.plot_policy(pi_star, env, title=f"π* — {tag} (VI)")

        
        def run_mc(seed):
            np.random.seed(seed)                  
            Q, _policy, ep_rewards = montecarlo_adv(
                env, gamma=GAMMA, epsilon=MC_EPSILON, total_num_episodes=MC_EPISODES)
            return Q, ep_rewards

        
        algos = {
            "Monte Carlo": run_mc,

            "SARSA": lambda sd: train_sarsa(
                env, gamma=GAMMA, seed=sd
            )[:2],

            "SARSA + Optuna": lambda sd: train_sarsa(
                env,
                episodes=5000,
                alpha=best_params["alpha"],
                gamma=best_params["gamma"],
                epsilon_start=best_params["epsilon_start"],
                epsilon_min=best_params["epsilon_min"],
                epsilon_decay=best_params["epsilon_decay"],
                seed=sd,
            )[:2],

            "Q-learning": lambda sd: train_q_learning(
                env, gamma=GAMMA, seed=sd
            )[:2],
        }


        curves = {}
        for name, fn in algos.items():
            runs, metrics = [], []
            for sd in SEEDS:
                Q, ep_rewards = fn(sd)
                runs.append(moving_average(ep_rewards))
                sr, ret, steps = evaluate_policy(Q, env)
                q_err = float(np.max(np.abs(Q - Q_star)))
                metrics.append((sr, ret, steps, q_err))
                rows.append(dict(env=tag, algo=name, seed=sd,
                                 success=sr, ret=ret, steps=steps, q_err=q_err))
            curves[name] = np.asarray(runs)
            show_policy(Q, f"π ({name}, {tag}, last seed)")   
            print(f"{name:13s} success={np.mean([m[0] for m in metrics]):.3f} "
                  f"return={np.mean([m[1] for m in metrics]):.4f} "
                  f"steps={np.mean([m[2] for m in metrics]):.2f} "
                  f"||Q-Q*||inf={np.mean([m[3] for m in metrics]):.4f}")

        plt.figure(figsize=(9, 5))
        for name, c in curves.items():
            m, s = c.mean(0), c.std(0)
            x = np.arange(len(m)) + WINDOW - 1
            plt.plot(x, m, label=name)
            plt.fill_between(x, m - s, m + s, alpha=0.2)
        plt.title(f"{tag}: Moving Average Reward ({WINDOW} eps, {len(SEEDS)} seeds)")
        plt.xlabel("Episode"); plt.ylabel("Mean Reward")
        plt.ylim(-1.05, 1.05); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUT}/curves_{tag}.png", dpi=150)
        plt.close()

    
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/summary.csv", index=False)
    print("\n *****SUMMARY (mean over seeds)*****")
    print(df.groupby(["env", "algo"])[["success", "ret", "steps", "q_err"]]
            .mean().round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in zip(axes, ["success", "ret"],
                              ["Ruccess Rate (eval greedy)", "Average return (eval greedy)"]):
        m = df.pivot_table(index="algo", columns="env", values=col)
        s = df.pivot_table(index="algo", columns="env", values=col, aggfunc="std")
        m.plot(kind="bar", yerr=s, ax=ax, capsize=4, rot=15)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{OUT}/final_performance.png", dpi=150)
    plt.close()
    print(f"\nFigures + CSVs saved in ./{OUT}/.")



if __name__ == "__main__":
    main()