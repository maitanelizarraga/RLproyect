import numpy as np
import matplotlib.pyplot as plt
import optuna

from env import GridworldEnv


def epsilon_greedy(Q, state, epsilon, n_actions, rng):
    """Explora con probabilidad epsilon; si no, elige una de las mejores acciones."""
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))

    # Si varias acciones tienen el mismo valor Q, se elige una al azar.
    best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
    return int(rng.choice(best_actions))


def train_sarsa(
    alpha,
    gamma,
    epsilon_start,
    epsilon_min,
    epsilon_decay,
    episodes,
    max_steps=200,
    is_slippery=False,
    seed=42,
):
    """
    Entrena SARSA y devuelve la tabla Q, las recompensas y los exitos por episodio.
    Epsilon disminuye progresivamente para pasar de explorar a explotar.
    """
    rng = np.random.default_rng(seed)
    env = GridworldEnv(render_mode=None, is_slippery=is_slippery)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    rewards_per_episode = []
    successes_per_episode = []

    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
            action = epsilon_greedy(Q, state, epsilon, env.n_actions, rng)
            total_reward = 0.0
            reached_goal = False

            for _ in range(max_steps):
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward

                if done:
                    # En un estado terminal no se suma valor futuro.
                    target = reward
                    Q[state, action] += alpha * (target - Q[state, action])
                    reached_goal = next_state == 3
                    break

                # SARSA utiliza la siguiente accion elegida por la politica actual.
                next_action = epsilon_greedy(
                    Q, next_state, epsilon, env.n_actions, rng
                )
                target = reward + gamma * Q[next_state, next_action]
                Q[state, action] += alpha * (target - Q[state, action])

                state, action = next_state, next_action

            rewards_per_episode.append(total_reward)
            successes_per_episode.append(1 if reached_goal else 0)
    finally:
        env.close()

    return Q, rewards_per_episode, successes_per_episode


def evaluate_policy(Q, episodes=100, max_steps=200, is_slippery=False, seed=10000):
    """Evalua la politica aprendida sin exploracion y devuelve exito y retorno medio."""
    rng = np.random.default_rng(seed)
    env = GridworldEnv(render_mode=None, is_slippery=is_slippery)
    successes = 0
    returns = []

    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0

            for _ in range(max_steps):
                # Politica greedy: se elige una de las acciones con mayor Q.
                best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
                action = int(rng.choice(best_actions))
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state

                if terminated or truncated:
                    if next_state == 3:
                        successes += 1
                    break

            returns.append(total_reward)
    finally:
        env.close()

    return successes / episodes, float(np.mean(returns))


def objective(trial):
    # Optuna prueba diferentes combinaciones de hiperparametros.
    alpha = trial.suggest_float("alpha", 0.01, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.80, 0.999)
    epsilon_start = trial.suggest_float("epsilon_start", 0.05, 1.0)
    epsilon_min = trial.suggest_float("epsilon_min", 0.01, 0.15)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.990, 0.9999)
    episodes = trial.suggest_int("episodes", 1000, 5000, step=500)

    # Evita una combinacion incoherente: epsilon minimo no supera el inicial.
    epsilon_min = min(epsilon_min, epsilon_start)
    is_slippery = False

    print(
        f"\n---> Iniciando Trial {trial.number} | "
        f"alpha: {alpha:.4f} | gamma: {gamma:.4f} | "
        f"epsilon inicial: {epsilon_start:.3f} | episodios: {episodes}"
    )

    Q, rewards, successes = train_sarsa(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        episodes=episodes,
        max_steps=200,
        is_slippery=is_slippery,
        seed=42,
    )

    # La evaluacion se hace sin exploracion, en episodios diferentes del entrenamiento.
    success_rate, mean_return = evaluate_policy(
        Q,
        episodes=100,
        max_steps=200,
        is_slippery=is_slippery,
        seed=10000,
    )

    trial.set_user_attr("mean_return", mean_return)
    trial.set_user_attr("training_mean_last_100", float(np.mean(rewards[-100:])))
    trial.set_user_attr(
        "training_success_rate_last_100", float(np.mean(successes[-100:]))
    )

    print(
        f"    Trial {trial.number} - Tasa de exito: {success_rate:.1%} "
        f"- Recompensa media: {mean_return:.3f}"
    )

    # Optuna maximiza la tasa de episodios que alcanzan la meta.
    return success_rate


def show_policy(Q):
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    wall_state = 5
    terminal_states = {3: "+1", 7: "-1"}

    print("\nPolitica aprendida:")
    for row in range(3):
        cells = []
        for col in range(4):
            state = row * 4 + col
            if state == wall_state:
                cells.append(" ### ")
            elif state in terminal_states:
                cells.append(f" {terminal_states[state]} ")
            else:
                cells.append(f"  {arrows[int(np.argmax(Q[state]))]}  ")
        print(" | ".join(cells))


def main():
    print("Iniciando optimizacion de SARSA con Optuna...")

    # TPE propone nuevas combinaciones basandose en los resultados anteriores.
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        study_name="optimizacion_sarsa_gridworld",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
        storage="sqlite:///optuna_sarsa.db",
    )

    n_trials = 50
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("OPTIMIZACION TERMINADA")
    print("=" * 60)
    print(f"Pruebas registradas: {len(study.trials)}")
    print(f"Mejor tasa de exito en evaluacion: {study.best_value:.1%}")
    print("\nMejores hiperparametros:")
    for name, value in study.best_params.items():
        print(f"    {name}: {value}")

    # Reentrena con la mejor configuracion encontrada por Optuna.
    best = study.best_params
    epsilon_min = min(best["epsilon_min"], best["epsilon_start"])

    Q, rewards, successes = train_sarsa(
        alpha=best["alpha"],
        gamma=best["gamma"],
        epsilon_start=best["epsilon_start"],
        epsilon_min=epsilon_min,
        epsilon_decay=best["epsilon_decay"],
        episodes=best["episodes"],
        max_steps=200,
        is_slippery=False,
        seed=2026,
    )

    final_success, final_return = evaluate_policy(Q, is_slippery=False, seed=20000)
    print(f"\nTasa de exito al reevaluar la mejor politica: {final_success:.1%}")
    print(f"Recompensa media de evaluacion: {final_return:.3f}")
    print(f"Tabla Q: {Q.shape[0]} estados x {Q.shape[1]} acciones")
    show_policy(Q)

    # Guarda la tabla Q y el historial de pruebas para reutilizarlos.
    np.save("Q_sarsa_optuna.npy", Q)
    study.trials_dataframe().to_csv("resultados_optuna_sarsa.csv", index=False)
    print("\nGuardado: Q_sarsa_optuna.npy")
    print("Guardado: resultados_optuna_sarsa.csv")
    print("Base de datos del estudio: optuna_sarsa.db")

    # Grafica la recompensa media movil del entrenamiento final.
    window = min(100, len(rewards))
    moving_average = np.convolve(
        rewards, np.ones(window) / window, mode="valid"
    )
    plt.figure(figsize=(9, 4))
    plt.plot(moving_average)
    plt.title(f"SARSA optimizado: recompensa media movil ({window} episodios)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa media")
    plt.tight_layout()
    plt.show()

    # Grafica el historial de optimizacion de Optuna.
    try:
        from optuna.visualization.matplotlib import plot_optimization_history

        plot_optimization_history(study)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"No se pudo mostrar la grafica de Optuna: {exc}")


if __name__ == "__main__":
    main()
