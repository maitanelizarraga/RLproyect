####
# MONTECARLO ADVANCED (in-class exercise)
####
import numpy as np
from env import GridworldEnv
import tools


def get_action(greedy_action, epsilon, n_actions):
    
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return greedy_action


def montecarlo_adv(env, gamma=0.999, epsilon=0.3, total_num_episodes=10000,
                   max_steps=200):
   
    P = env.P
    n_states, n_actions = len(P), len(P[0])

    Q = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)
    returns = np.zeros((n_states, n_actions))
    N = np.zeros((n_states, n_actions))
    episode_rewards = []

    for ep_num in range(total_num_episodes):
        state, _ = env.reset()
        trajectory = []
        for _ in range(max_steps):
            action = get_action(policy[state], epsilon, n_actions)
            s_prime, rwd, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, rwd))
            state = s_prime
            if terminated or truncated:
                break
        episode_rewards.append(sum(r for _, _, r in trajectory))

        # Calcula G_t para cada instante de la trayectoria, de atrás hacia delante.
        # Después recorremos la trayectoria en orden cronológico para que sea
        # realmente first-visit MC (primera visita a cada par estado-acción).
        returns_from_t = [0.0] * len(trajectory)
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            _, _, rwd = trajectory[t]
            G = rwd + gamma * G
            returns_from_t[t] = G

        visited = set()
        for t, (s, a, _rwd) in enumerate(trajectory):
            if (s, a) in visited:
                continue
            visited.add((s, a))
            G = returns_from_t[t]
            returns[s, a] += G
            N[s, a] += 1
            Q[s, a] = returns[s, a] / N[s, a]
            policy[s] = int(np.argmax(Q[s]))

    return Q, policy, episode_rewards


def get_value_function(Q, policy):
    """V(s) = Q(s, pi(s))."""
    return np.array([Q[s, policy[s]] for s in range(len(policy))])


if __name__ == "__main__":
    env = GridworldEnv(is_slippery=False)
    Q, policy, rewards = montecarlo_adv(env, gamma=0.8, epsilon=0.3,
                                        total_num_episodes=10000)
    V = get_value_function(Q, policy)
    tools.plot_value_function(V, env)
    tools.plot_policy(policy, env)