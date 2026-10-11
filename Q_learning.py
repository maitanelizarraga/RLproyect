
import numpy as np
import matplotlib.pyplot as plt

import tools
from env import GridworldEnv
from td_learning import (get_action, epsilon_at, evaluate_policy, greedy_V,
                          show_policy, show_environment, play_greedy, GOAL_STATE)


def train_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
                     epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                     max_steps=200, seed=42):
    """Trains Q-learning on `env`. Returns (Q, rewards, successes)."""
    np.random.seed(seed)
    nA = env.n_actions
    Q = np.zeros((env.n_states, nA))
    rewards, successes = [], []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        eps = epsilon_at(ep, epsilon_start, epsilon_min, epsilon_decay)
        total, hit = 0.0, False

        
        for _ in range(max_steps):
            a = get_action(int(np.argmax(Q[s])), eps, nA)
            s2, r, term, trunc, _ = env.step(a)
            total += r

            # No se añade valor futuro si se alcanza un estado terminal.
            future = 0.0 if (term or trunc) else float(np.max(Q[s2]))
            Q[s, a] += alpha * (r + gamma * future - Q[s, a])

            if term or trunc:
                hit = (s2 == GOAL_STATE)
                break

            s = s2

        rewards.append(total)
        successes.append(int(hit))

    return Q, rewards, successes


if __name__ == "__main__":
    env = GridworldEnv(is_slippery=False)          
    Q, rewards, successes = train_q_learning(env, episodes=5000, alpha=0.1, gamma=0.99)

    sr, ret, steps = evaluate_policy(Q, env)
    print(f"Q-learning: success={sr:.1%}  return={ret:.4f}  steps={steps:.2f}")
    show_policy(Q, "Q-learning")

    tools.plot_policy(np.argmax(Q, axis=1), env, title="π(s) — Q-learning")
    tools.plot_value_function(greedy_V(Q), env, title="V(s) — Q-learning")

    window = 100
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(9, 4))
    plt.plot(ma)
    plt.title(f"Q-learning: recompensa media móvil ({window} episodios)")
    plt.xlabel("Episodio"); plt.ylabel("Recompensa media")
    plt.tight_layout(); plt.show()