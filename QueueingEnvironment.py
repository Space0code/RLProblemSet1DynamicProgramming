import numpy as np


class QueueingEnvironment:
    """Controlled queue MDP from Problem Set 1.

    States:  queue lengths 0, 1, ..., N-1
    Actions: 0 = low service rate, 1 = high service rate
    """

    # Actions are numbered so they can be used directly as array indices
    LOW = 0
    HIGH = 1

    def __init__(self, N=100, p=0.5, q_low=0.51, q_high=0.6,
                 cost_low=0.0, cost_high=0.01, gamma=0.9):
        self.N = N
        self.states = np.arange(N)    # 0, 1, ..., N-1
        self.p = p                    # arrival probability
        self.gamma = gamma            # discount factor

        # Per-action values, indexed by action: 
        self.n_actions = 2
        self.q = np.array([q_low, q_high])           # service probability
        self.cost = np.array([cost_low, cost_high])  # action cost c(a)

        # reward
        self.R = -(self.states / N) ** 2 - self.cost[:, None] # (2, N) -> (action, state)

        # transition P
        # arrival I and service S are independent: +1 if I=1,S=0; -1 if I=0,S=1; stay otherwise.
        # trunc keeps the queue in [0, N-1], so at the edges two outcomes land on the same state (hence +=)
        def trunc(x):
            return min(N-1, max(x,0))

        P = np.zeros((self.n_actions, N, N))
        for a in range(self.n_actions):
            increase = self.p * (1-self.q[a])
            decrease = (1-self.p) * self.q[a]
            for x in range(N):
                P[a, x, trunc(x+1)] += increase 
                P[a, x, trunc(x-1)] += decrease
                P[a, x, x] += 1 - increase - decrease

        self.P = P

    def policy_model(self, policy):
        """Reward vector and transition matrix of a fixed policy (the Markov chain it induces).

        Input:
            policy: int array of shape (N,), policy[x] = action taken in state x.
        Returns:
            r_pi: shape (N,),    r_pi[x]    = R[policy[x], x]
            P_pi: shape (N, N),  P_pi[x, :] = P[policy[x], x, :]

        """
        r_pi = np.zeros(self.N)
        P_pi = np.zeros((self.N, self.N))

        for x in range(self.N):
            r_pi[x] = self.R[policy[x], x]
            P_pi[x, :] = self.P[policy[x], x, :]


        return r_pi, P_pi
