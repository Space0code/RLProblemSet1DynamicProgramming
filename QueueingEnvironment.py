# STEP 1 — Build the MDP model (P and r as numpy arrays).
#
# Constants from the PDF:
#   N = 100, p = 0.5, q_low = 0.51, q_high = 0.6, c_low = 0, c_high = 0.01, gamma = 0.9
#   Actions: index 0 = a_low, index 1 = a_high  (store q and c as arrays of length 2)
#
# KEY INSIGHT (transition): I_t and S_t are independent coin flips, so from state x with action a:
#   go UP   (+1)  when I=1, S=0  -> prob ?
#   go DOWN (-1)  when I=0, S=1  -> prob ?
#   STAY    ( 0)  otherwise      -> prob 1 - up - down
#   Then truncate the next state to [0, N-1]. At x=0 "down" lands on 0, at x=N-1 "up" lands on N-1.
#   Use += when filling P so boundary probabilities accumulate correctly.
#
# Sanity check: every row P[a, x, :] must sum to 1.

import numpy as np


class QueueingMDP:
    def __init__(self, N=100, p=0.5, q=(0.51, 0.6), c=(0.0, 0.01), gamma=0.9):
        # TODO: store parameters, then call the builders below
        pass

    def build_transitions(self):
        # TODO: return P with shape (2, N, N), P[a, x, x'] = P(x' | x, a)
        pass

    def build_rewards(self):
        # TODO: return R with shape (N, 2), R[x, a] = -(x/N)**2 - c[a]
        # Hint: vectorize with np.arange(N)
        pass
