########
### SIMPLE POLICY ITERATION IMPLEMENTATION
########

import matplotlib.pyplot as plt
import numpy as np
from env import GridworldEnv

def policy_iteration(P, gamma=0.999, epsilon=1e-8):
    # set all required variables (fixed for all iterations)
    #NOTE: gamma and epsilon set through the args
    n_states = len(P)  # check env.P in gymnasium to get an idea of the structure or evaluate in debugger
    n_actions = len(P[0])

    # set all required variables (initialization)
    delta = 0.0  
    k_iter = 0  # just to record iterations
    j_iter = 0
    V_pi = np.zeros(n_states)  # we need 2 value functions (k and k+1) in the same algorithm
    V_pi_new = np.zeros(n_states)
    Q_pi = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states, dtype=int)  # NOTE: why not use 2D with values being each action probs? Gymnasium compatibility and because optimal policy always has a single action (no probs, only the action number)
    policy_new = np.zeros(n_states, dtype=int)  # NOTE: not the most efficient memory-wise. Totally possible to only have one policy throughout the algorithm instead of 2
  
    while True:  # k repetitions, outer loop
        # TODO: POLICY EVALUATION section
        while True:  # j repetitions, inner loop
            for s in P:
                a = policy[s]  # NOTE: as we will later use argmax(Q), it will always be deterministic in the algorithm
                for prob, s_prime, reward, done in P[s][a]:
                    # NOTE: make sure V(s')=0 for terminal states
                    future_value = 0.0 if done else V_pi[s_prime]
                    V_pi_new[s] += prob * (reward + gamma * future_value)

            delta = np.max(np.abs(V_pi_new - V_pi))
            j_iter += 1
            if delta <= epsilon:  # exit loop
                break
            else:
                V_pi = V_pi_new.copy()  # NOTE: avoid variable pointing to same object reference!
                V_pi_new = np.zeros(n_states)

        # TODO: POLICY IMPROVEMENT section
        for s in P:
            for a in P[s]:
                for prob, s_prime, reward, done in P[s][a]:
                    future_value = 0.0 if done else V_pi_new[s_prime]
                    Q_pi[s][a] += prob * (reward + gamma * future_value)
            a = np.argmax(Q_pi[s])
            policy_new[s] = a

        k_iter += 1

        if np.array_equal(policy_new, policy):
            break  # NOTE: in this case previous V_pi is also V*(s), both policies are the same
        else:
            policy = policy_new.copy()
            policy_new = np.zeros(n_states, dtype=int)
            Q_pi = np.zeros((n_states, n_actions))
            V_pi = V_pi_new.copy()
            V_pi_new = np.zeros(n_states)

    return V_pi_new, Q_pi, policy_new


def generate_graph(value_history):
    """Generate convergence graph of state values across outer Policy Iteration steps."""
    name_list = (
        "(1,3)",
        "(2,3)",
        "(3,3)",
        "+1 Goal",
        "(1,2)",
        "Wall (#)",
        "(3,2)",
        "-1 Pit",
        "(1,1)",
        "(2,1)",
        "(3,1)",
        "(4,1)",
    )

    color_list = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "black",  # Goal line
        "tab:purple",
        "gray",  # Wall line
        "tab:cyan",
        "tab:red",  # Pit line
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "mediumpurple",
    )

    iterations = np.arange(len(value_history))

    plt.figure(figsize=(10, 6))

    for state in range(12):
        state_values = [v[state] for v in value_history]
        linestyle = "--" if state in (3, 5, 7) else "-"
        plt.plot(
            iterations,
            state_values,
            color=color_list[state],
            label=name_list[state],
            linestyle=linestyle,
            linewidth=2 if state not in (3, 5, 7) else 1,
            marker="o",  # Markers help highlight the few outer steps
        )

    plt.legend(
        title="Coordinates (x, y)\n(Left → Right, Bottom → Up)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    plt.xlim((0, len(value_history) - 1))
    plt.ylabel("True Policy Return $V^{\\pi_k}(s)$", fontsize=13)
    plt.xlabel("Outer Policy Iteration ($k$)", fontsize=13)
    plt.title("Policy Iteration State Convergence", fontsize=14, pad=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    # plt.savefig("./pi_output.jpg", dpi=300)
    plt.show()

if __name__ == "__main__":
    import tools
    import gymnasium as gym

    env = GridworldEnv(is_slippery=False)
    P = env.P

    V, Q, policy = policy_iteration(P, gamma=0.8, epsilon=1e-8)

    # NOTE: check resulting value function and policy to evaluate correctness
    tools.plot_value_function(V, env)
    tools.plot_policy(policy, env)

    # TODO: repeat with slippery environment (stochastic transitions)
    env = GridworldEnv(is_slippery=True)

    # TODO: repeat with gymnasium frozenlake env
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    P = env.unwrapped.P  # required in gymnasium

    # TODO: repeat with gymnasium slippery frozenlake env
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")
    P = env.unwrapped.P
