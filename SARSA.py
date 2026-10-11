import numpy as np
import matplotlib.pyplot as plt
from env import GridworldEnv


def epsilon_greedy(Q, state, epsilon, n_actions):
    """Elige una acción aleatoria con probabilidad epsilon; si no, la mejor conocida."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))
    return int(np.random.choice(best_actions))

def train_sarsa(episodes=5000,alpha=0.1,gamma=0.99,epsilon=0.1,max_steps=200,is_slippery=False,seed=42,):
    """Entrena SARSA sobre el Gridworld definido en env.py."""
    np.random.seed(seed)
    environment = GridworldEnv(render_mode=None, is_slippery=is_slippery)
    n_states = environment.n_states
    n_actions = environment.n_actions
    Q = np.zeros((n_states, n_actions), dtype=float)
    episode_rewards = []

    for episode in range(episodes):
        state, episode = environment.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        total_reward = 0.0

        for episode in range(max_steps):
            next_state, reward, terminated, truncated, episode = environment.step(action)
            total_reward += reward
            done = terminated or truncated

            if done:
                Q[state, action] += alpha * (reward - Q[state, action])
                break

            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)
            target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])
            state, action = next_state, next_action

        episode_rewards.append(total_reward)

    environment.close()
    return Q, episode_rewards

def show_policy(Q):
    """Imprime la acción greedy de cada estado."""
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    wall_state = 5
    terminal_states = {3: "+1", 7: "-1"}

    print("\nPolítica aprendida:")
    for row in range(3):
        cells = []
        for col in range(4):
            state = row * 4 + col
            if state == wall_state:
                cells.append("###")
            elif state in terminal_states:
                cells.append(f" {terminal_states[state]} ")
            else:
                cells.append(f" {arrows[int(np.argmax(Q[state]))]} ")
        print(" | ".join(cells))

def show_environment():
    """Muestra el entorno """
    environment = GridworldEnv(render_mode="rgb_array")
    environment.reset()
    image = environment.render()

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title("Gridworld")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    environment.close()

if __name__ == "__main__":
    Q, rewards = train_sarsa(
        episodes=5000,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        is_slippery=False,
    )
    

    print("Entrenamiento SARSA terminado.")
    print(f"Tabla Q: {Q.shape[0]} estados x {Q.shape[1]} acciones")
    show_policy(Q)

    window = 100
    if len(rewards) >= window:
        moving_average = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        plt.figure(figsize=(9, 4))
        plt.plot(moving_average)
        plt.title(f"Recompensa media móvil ({window} episodios)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa media")
        plt.tight_layout()
        plt.show()

    show_environment()
