
import numpy as np
import matplotlib.pyplot as plt
import optuna
import json
from pathlib import Path


from env import GridworldEnv
from SARSA import train_sarsa
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


def main():
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

    print("Hiperparámetros guardados en best_sarsa_params.json")

    Q, rewards, successes = train_sarsa(
        TUNE_ENV, episodes=FIXED_EPISODES, alpha=p["alpha"], gamma=p["gamma"],
        epsilon_start=p["epsilon_start"],
        epsilon_min=min(p["epsilon_min"], p["epsilon_start"]),
        epsilon_decay=p["epsilon_decay"], seed=2026)

    sr, ret, _ = evaluate_policy(Q, TUNE_ENV, episodes=500, seed=20_000)
    print(f"\nRe-evaluación: success={sr:.1%}  return={ret:.4f}")
    show_policy(Q, "SARSA (Optuna, stochastic)")

    study.trials_dataframe().to_csv("resultados_optuna_sarsa_stochastic.csv", index=False)

    window = 100
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(9, 4))
    plt.plot(ma)
    plt.title(f"SARSA optimizado: recompensa media móvil ({window})")
    plt.xlabel("Episodio"); plt.ylabel("Recompensa media")
    plt.tight_layout(); plt.show()

    try:
        from optuna.visualization.matplotlib import plot_optimization_history
        plot_optimization_history(study)
        plt.tight_layout(); plt.show()
    except Exception as exc:
        print(f"No se pudo mostrar la gráfica de Optuna: {exc}")


if __name__ == "__main__":
    main()