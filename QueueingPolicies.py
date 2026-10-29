# STEP 2 — Policies + Policy Evaluation (Problem 1)
# STEP 3 — Value Iteration and Policy Iteration (Problem 2)
#
# Represent a deterministic policy as an int array of length N with entries 0 (low) or 1 (high).

import numpy as np


# ---------- Problem 1 ----------

def lazy_policy(N):
    # TODO: always action 0
    pass


def aggressive_policy(N, threshold=50):
    # TODO: action 0 for x < threshold, action 1 for x >= threshold
    pass


def policy_evaluation(mdp, policy):
    # TODO: direct solve  V = (I - gamma * P_pi)^-1 r_pi
    #   P_pi[x, :] = mdp.P[policy[x], x, :]   -> fancy indexing: mdp.P[policy, np.arange(N)]
    #   r_pi[x]    = mdp.R[x, policy[x]]
    # Use np.linalg.solve(A, b) instead of inverting the matrix.
    pass


# ---------- Problem 2 ----------

def q_values(mdp, V):
    # TODO: return Q with shape (N, 2): Q[x, a] = R[x, a] + gamma * sum_x' P[a, x, x'] V[x']
    # Hint: (mdp.P @ V) has shape (2, N) -> transpose it.
    pass


def value_iteration(mdp, n_iter, V0=None):
    # TODO: start from V = zeros, repeat n_iter times: V = max_a Q(x, a)
    # Return final V, greedy policy (argmax), and the list of V after each iteration
    # (you need snapshots at 10, 20, 50, 100 for the plots).
    pass


def policy_iteration(mdp, n_iter, policy0=None):
    # TODO: start from some policy (e.g. lazy), repeat n_iter times:
    #   V = policy_evaluation(mdp, policy)      # evaluation
    #   policy = argmax_a Q(x, a) under V       # greedy improvement
    # Same return format as value_iteration.
    # Note: PI usually converges in a handful of iterations — that's expected.
    pass
