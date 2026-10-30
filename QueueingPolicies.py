"""Policies and dynamic-programming algorithms for the queueing MDP.

Conventions used everywhere:
    policy : int array, shape (N,), policy[x] in {env.LOW, env.HIGH}
    V      : float array, shape (N,), V[x] = value of state x
    Q      : float array, shape (n_actions, N), Q[a, x]
"""
import numpy as np


def lazy_policy(env):
    return np.full(env.N, env.LOW) # always low

def aggressive_policy(env, threshold=50):
    return np.where(env.states >= threshold, env.HIGH, env.LOW) # LOW for x < threshold, HIGH for x >= threshold

def evaluate_policy(env, policy):
    """
    V_pi = (I - gamma * P_pi)^-1 r_pi.

    directly solving the Bellman equations
    """
    r_pi, P_pi = env.policy_model(policy)
    A = np.eye(env.N) - env.gamma * P_pi    
    return np.linalg.solve(A, r_pi)          


def q_values(env, V):
    """
    Q[a, x] = R[a, x] + gamma * sum_y P[a, x, y] * V[y].
    """
    return env.R + env.gamma * (env.P @ V)


def greedy_policy(env, V):
    """The action with the highest Q-value in each state."""
    return q_values(env, V).argmax(axis=0)


def value_iteration(env, n_iters, record_at=()):
    """Run n_iters sweeps of V <- max_a Q(V), starting from V = 0.
    
    V_k+1 = max(Q(a,x))

    Returns:
        V         : value after n_iters sweeps
        snapshots : dict {k: copy of V after sweep k} for every k in record_at
    """

    V = np.zeros(env.N)
    snapshots = {}
    for k in range(1, n_iters + 1):
        q = q_values(env, V)
        V = q.max(axis=0)                        # max over actions of q_values(env, V)

        if k in record_at:
            snapshots[k] = V.copy()
    return V, snapshots


def policy_iteration(env, n_iters, record_at=()):
    """Alternate evaluation and greedy improvement, starting from the lazy policy.

    One iteration = evaluate the current policy, then improve it.
    "Value after k iterations" = the V evaluated in iteration k.
    By the policy improvement theorem V never gets worse, and once the policy
    stops changing it is optimal and V = V*.

    Runs all n_iters iterations even after the policy is stable, so that the
    runtime of a fixed number of iterations can be compared with value iteration.

    Returns:
        V         : value of the last evaluated policy
        policy    : the last improved policy
        snapshots : dict {k: copy of V after iteration k} for every k in record_at
    """

    policy = lazy_policy(env)

    snapshots = {}
    for k in range(1,n_iters+1):
        # policy evaluation: exact V of the current policy

        V = evaluate_policy(env, policy)

        if k in record_at:
            snapshots[k] = V.copy()

        # policy improvement: greedy w.r.t. V
        policy = greedy_policy(env, V)

    return V, policy, snapshots
