########
### SIMPLE VALUE ITERATION IMPLEMENTATION
########

import matplotlib.pyplot as plt
import numpy as np
from env import GridworldEnv

def value_iteration(P, gamma=0.999, epsilon=1e-8):
    """
    Value iteration algorithm
    Inputs:
      env: whole gymnasium or basic env to extract multiple required parameters
      gamma: discount factor
      epsilon: desired number for deciding convergence
    Ouputs:
      V_new: last optimal value-function
      Q_new: last optimal Q-function
      policy: optimal policy extracted by Q-function
      value_history: list of all appended V variables (appended at each iteration)
    """
    # set all required variables (fixed for all iterations)
    #NOTE: gamma and epsilon set through the args
    n_states = len(P)  # check env.P in gymnasium to get an idea of the structure or evaluate in debugger
    n_actions = len(P[0])

    # set all required variables (initialization)
    delta = 0.0  
    k_iter = 0  # just to record iterations
    V = np.zeros(n_states)  # we need 2 value functions (k and k+1) in the same algorithm
    V_new = np.zeros(n_states)
    Q_new = np.zeros((n_states, n_actions))
    policy = np.zeros(n_states)  # NOTE: why not use 2D with values being each action probs? Gymnasium compatibility and because optimal policy always has a single action (no probs, only the action number)
  
    # list of V(s) values throughout iterations (extrictly not required for algorithm, useful for getting more intuition with my auxiliary functions)
    value_history = []
    value_history.append(V)

    # TODO: general loop to get V*(s), Q*(s,a)
    while True:
        # V = V_new.copy()
        for s in P:
            for a in P[s]:
                for prob, s_prime, reward, done in P[s][a]:
                    # NOTE: make sure V(s')=0 for terminal states
                    future_value = 0.0 if done else V[s_prime]  
                    Q_new[s][a] += prob * (reward + gamma * future_value)

            V_new[s] = max(Q_new[s])
        
        value_history.append(V_new)
        delta = np.max(np.abs(V_new - V))
        k_iter += 1

        if delta <= epsilon:  # exit loop
            break
        else: # prepare for next iteration
            V = V_new.copy()  # NOTE: avoid variable pointing to same object reference!
            V_new = np.zeros(n_states)
            Q_new = np.zeros((n_states, n_actions))

    # TODO: extract optimal policy from Q*(s,a)
    for s in range(n_states):
        a = np.argmax(Q_new[s]) 
        policy[s] = a

    return V_new, Q_new, policy, value_history

def generate_graph(value_history):
    """
    Auxiliary function to generate graph of state-values for each iteration. No need to check.
    Only works on basic env
    """
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
        # Use dashed lines for terminal/wall states so non-terminal curves stay clear
        linestyle = (
            "--" if state in (3, 5, 7) else "-"
        )  # Goal (3), Wall (5), Pit (7)
        plt.plot(
            iterations,
            state_values,
            color=color_list[state],
            label=name_list[state],
            linestyle=linestyle,
            linewidth=2 if state not in (3, 5, 7) else 1,
        )

    # Place legend OUTSIDE the axes to avoid hiding data
    plt.legend(
        title="Coordinates (x, y)\n(Left → Right, Bottom → Up)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    # plt.ylim((-1.1, +1.1))
    plt.xlim((0, len(value_history)-1))
    plt.ylabel("Value $V(s)$", fontsize=13)
    plt.xlabel("Iteration", fontsize=13)
    plt.title("Value Iteration State Convergence", fontsize=14, pad=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    # plt.savefig("./output.jpg", dpi=300)
    plt.show()

if __name__ == "__main__":
    import tools
    import gymnasium as gym

    env = GridworldEnv(is_slippery=False)
    P = env.P

    V, Q, policy, value_history = value_iteration(P, gamma=0.8, epsilon=1e-8)

    # NOTE: check resulting value function and policy to evaluate correctness
    tools.plot_value_function(V, env)
    tools.plot_policy(policy, env)

    # NOTE: generate graph of convergence for each value-state throughout algorithm iterations
    generate_graph(value_history)


    # TODO: repeat with slippery environment (stochastic transitions)
    env = GridworldEnv(is_slippery=True)

    # TODO: repeat with gymnasium frozenlake env
    env = gym.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    P = env.unwrapped.P  # required in gymnasium

    # TODO: repeat with gymnasium slippery frozenlake env
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")
    P = env.unwrapped.P
