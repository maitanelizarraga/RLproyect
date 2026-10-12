
import numpy as np
import matplotlib.pyplot as plt
import optuna
import json
from pathlib import Path


from env import GridworldEnv
from SARSA import train_sarsa
from Q_learning import train_q_learning
from td_learning import evaluate_policy, show_policy

TUNE_ENV = GridworldEnv(is_slippery=True)   
FIXED_EPISODES = 3000


def objective(trial):
    alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    epsilon_start = trial.suggest_float("epsilon_start", 0.3, 1.0)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.10)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.995, 0.9999)
    eps_min = min(epsilon_min, epsilon_start)      

    Q, rewards, successes = train_sarsa(
        TUNE_ENV, episodes=FIXED_EPISODES, alpha=alpha, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=eps_min,
        epsilon_decay=epsilon_decay, max_steps=200, seed=30)

    success_rate, mean_return, _ = evaluate_policy(
        Q, TUNE_ENV, episodes=500, max_steps=200, seed=20_000)

    trial.set_user_attr("success_rate", success_rate)
    print(f"  Trial {trial.number:3d}: success={success_rate:.1%}  return={mean_return:.4f}")
    return mean_return



def objective_q_learning(trial):
    alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    epsilon_start = trial.suggest_float("epsilon_start", 0.3, 1.0)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.10)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.995, 0.9999)

    eps_min = min(epsilon_min, epsilon_start)

    Q, rewards, successes = train_q_learning(
        TUNE_ENV,
        episodes=FIXED_EPISODES,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=eps_min,
        epsilon_decay=epsilon_decay,
        max_steps=200,
        seed=30
    )

    success_rate, mean_return, _ = evaluate_policy(
        Q, TUNE_ENV, episodes=500,
        max_steps=200, seed=20_000
    )

    trial.set_user_attr("success_rate", success_rate)
    print(
        f"Trial {trial.number:3d}: "
        f"success={success_rate:.1%} return={mean_return:.4f}"
    )

    return mean_return


def optimize_q_learning():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=30)
    )

    study.optimize(objective_q_learning, n_trials=50)

    print("\nMejor resultado de Q-learning:", study.best_value)
    print("Mejores hiperparámetros:", study.best_params)

    p = study.best_params
    best_params = {
        **p,
        "epsilon_min": min(p["epsilon_min"], p["epsilon_start"])
    }

    with open("best_q_learning_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    Q, rewards, successes = train_q_learning(
        TUNE_ENV,
        episodes=FIXED_EPISODES,
        alpha=p["alpha"],
        gamma=p["gamma"],
        epsilon_start=p["epsilon_start"],
        epsilon_min=best_params["epsilon_min"],
        epsilon_decay=p["epsilon_decay"],
        max_steps=200,
        seed=2026
    )

    sr, ret, _ = evaluate_policy(
        Q, TUNE_ENV, episodes=500,
        max_steps=200, seed=20_000
    )

    print(f"Reevaluación Q-learning: success={sr:.1%} return={ret:.4f}")
    show_policy(Q, "Q-learning (Optuna, stochastic)")

    study.trials_dataframe().to_csv(
        "results_optuna_q_learning_stochastic.csv",
        index=False
    )


def main():
    #  SARSA 
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=30)
    )
    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 60)
    print(f"Best return: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    p = study.best_params

    
    # Guardar los mejores hiperparámetros para run_experiments.py
    best_params = {
        **p,
        "epsilon_min": min(p["epsilon_min"], p["epsilon_start"]),
    }

    with open("best_sarsa_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)

    print("Saved hyperparameters in best_sarsa_params.json")

    Q, rewards, successes = train_sarsa(
        TUNE_ENV, episodes=FIXED_EPISODES, alpha=p["alpha"], gamma=p["gamma"],
        epsilon_start=p["epsilon_start"],
        epsilon_min=min(p["epsilon_min"], p["epsilon_start"]),
        epsilon_decay=p["epsilon_decay"], seed=2026)

    sr, ret, _ = evaluate_policy(Q, TUNE_ENV, episodes=500, seed=20_000)
    print(f"\nRe-evaluación: success={sr:.1%}  return={ret:.4f}")
    show_policy(Q, "SARSA (Optuna, stochastic)")

    study.trials_dataframe().to_csv("results_optuna_sarsa_stochastic.csv", index=False)

    window = 100
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(9, 4))
    plt.plot(ma)
    plt.title(f"SARSA optimized: Moving average Reward ({window})")
    plt.xlabel("Episode"); plt.ylabel("Mean reward")
    plt.tight_layout(); plt.show()

    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(study)
        plt.tight_layout(); plt.show()
    except Exception as exc:
        print(f"Can not show the Optuna graphic: {exc}")
    
    #  Q-LEARNING 

    print("\n" + "=" * 60)
    print("OPTIMIZACIÓN DE Q-LEARNING")
    print("=" * 60)

    study_q = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=30)
    )
    study_q.optimize(objective_q_learning, n_trials=50)

    print("\n" + "=" * 60)
    print(f"Best return: {study_q.best_value:.4f}")
    print("Best params:")
    for k, v in study_q.best_params.items():
        print(f"    {k}: {v}")

    p_q = study_q.best_params

    # Guardar los mejores hiperparámetros para run_experiments.py
    best_params_q = {
        **p_q,
        "epsilon_min": min(p_q["epsilon_min"], p_q["epsilon_start"]),
    }

    with open("best_q_learning_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_q, f, indent=4)

    print("Saved hyperparameters in best_q_learning_params.json")

    Q, rewards, successes = train_q_learning(
        TUNE_ENV,
        episodes=FIXED_EPISODES,
        alpha=p_q["alpha"],
        gamma=p_q["gamma"],
        epsilon_start=p_q["epsilon_start"],
        epsilon_min=min(p_q["epsilon_min"], p_q["epsilon_start"]),
        epsilon_decay=p_q["epsilon_decay"],
        max_steps=200,
        seed=2026
    )

    sr, ret, _ = evaluate_policy(
        Q, TUNE_ENV, episodes=500, max_steps=200, seed=20_000
    )
    print(f"\nRe-evaluación: success={sr:.1%}  return={ret:.4f}")
    show_policy(Q, "Q-learning (Optuna, stochastic)")

    study_q.trials_dataframe().to_csv(
        "results_optuna_q_learning_stochastic.csv", index=False
    )

    window = 100
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(9, 4))
    plt.plot(ma)
    plt.title(f"Q-learning optimized: Moving average Reward ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Mean reward")
    plt.tight_layout()
    plt.show()

    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(study_q)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Can not show the Optuna graphic: {exc}")


if __name__ == "__main__":
    main()