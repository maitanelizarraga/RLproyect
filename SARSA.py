
import numpy as np
import matplotlib.pyplot as plt

import tools
from env import GridworldEnv
from td_learning import (get_action, epsilon_at, evaluate_policy, greedy_V,
                          show_policy, show_environment, play_greedy, GOAL_STATE)


def train_sarsa(env, episodes=5000, alpha=0.1, gamma=0.99,
                epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.999,
                max_steps=200, seed=42):
    """Trains SARSA on `env`. Returns (Q, rewards_per_episode, successes_per_episode)."""
    np.random.seed(seed)
    nA = env.n_actions
    Q = np.zeros((env.n_states, nA))
    rewards, successes = [], []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)          
        eps = epsilon_at(ep, epsilon_start, epsilon_min, epsilon_decay)
        a = get_action(int(np.argmax(Q[s])), eps, nA)
        total, hit = 0.0, False

                
        for _ in range(max_steps):
            s2, r, term, trunc, _ = env.step(a)
            total += r

            if term or trunc:
                Q[s, a] += alpha * (r - Q[s, a])
                hit = (s2 == GOAL_STATE)
                break

            a2 = get_action(int(np.argmax(Q[s2])), eps, nA)
            Q[s, a] += alpha * (
                r + gamma * Q[s2, a2] - Q[s, a]
            )
            s, a = s2, a2


        rewards.append(total)
        successes.append(int(hit))

    return Q, rewards, successes


if __name__ == "__main__":
    env = GridworldEnv(is_slippery=False)          
    Q, rewards, successes = train_sarsa(env, episodes=5000, alpha=0.1, gamma=0.99)

    sr, ret, steps = evaluate_policy(Q, env)
    print(f"SARSA: success={sr:.1%}  return={ret:.4f}  steps={steps:.2f}")
    print(f"Tabla Q: {Q.shape[0]} estados x {Q.shape[1]} acciones")
    show_policy(Q, "SARSA")

    tools.plot_policy(np.argmax(Q, axis=1), env, title="π(s) — SARSA")
    tools.plot_value_function(greedy_V(Q), env, title="V(s) — SARSA")

    window = 100
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(9, 4))
    plt.plot(ma)
    plt.title(f"SARSA: recompensa media móvil ({window} episodios)")
    plt.xlabel("Episodio"); plt.ylabel("Recompensa media")
    plt.tight_layout(); plt.show()

    show_environment(GridworldEnv(render_mode="rgb_array", is_slippery=False))
    
    