
import time
import numpy as np

GOAL_STATE = 3  
ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def get_action(greedy_action, epsilon, n_actions):
    """eps-greedy: random action with prob eps, otherwise the greedy one."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return greedy_action


def epsilon_at(episode, eps_start=1.0, eps_min=0.05, eps_decay=0.999):
    """Decaying exploration: explore early, greedy in the limit."""
    return max(eps_min, eps_start * (eps_decay ** episode))


def evaluate_policy(Q, env, episodes=200, max_steps=200, gamma=0.99, seed=10_000):
    """Evalúa la política greedy sin exploración. Devuelve éxito, retorno descontado y duración media.

    La duración incluye todos los episodios, incluso los que terminan en el pozo
    o llegan al límite de pasos. El retorno se descuenta porque la recompensa
    solo aparece al llegar a una casilla terminal (+1 o -1)."""
    np.random.seed(seed)
    successes, returns, steps = 0, [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        G, n = 0.0, 0
        for _ in range(max_steps):
            s2, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
            G += (gamma ** n) * r        
            n += 1
            if term or trunc:
                break
            s = s2
        returns.append(G)
        steps.append(n)
        if s2 == GOAL_STATE:
            successes += 1
    mean_steps = float(np.mean(steps)) if steps else float("nan")
    return successes / episodes, float(np.mean(returns)), mean_steps


def greedy_V(Q):
    """V(s) = max_a Q(s,a) — for tools.plot_value_function."""
    return np.max(Q, axis=1)


def show_policy(Q, title="Política aprendida"):
    """ASCII policy print (kept from the original SARSA.py)."""
    print(f"\n{title}:")
    for row in range(3):
        cells = []
        for col in range(4):
            s = row * 4 + col
            if s == 5:
                cells.append(" ### ")
            elif s == 3:
                cells.append(" +1 ")
            elif s == 7:
                cells.append(" -1 ")
            else:
                cells.append(f"  {ARROWS[int(np.argmax(Q[s]))]}  ")
        print(" | ".join(cells))


def show_environment(env):
    """Static rgb_array render — kept from the original SARSA.py."""
    import matplotlib.pyplot as plt
    env.reset()
    plt.figure(figsize=(8, 6))
    plt.imshow(env.render())
    plt.title("Gridworld")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    env.close()


def play_greedy(Q, env, delay=0.35, max_steps=100):
    """One greedy episode in the pygame window. env must be render_mode='human'."""
    s, _ = env.reset()
    for _ in range(max_steps):
        s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        time.sleep(delay)
        if term or trunc:
            break
    time.sleep(1.0)
    env.close()