# STEP 4 — Run experiments and make plots for the report.
#
# Problem 1:
#   [ ] build mdp, evaluate lazy & aggressive
#   [ ] plot V_lazy - V_aggr over x; print values at x=50 and x=80
#       (positive => lazy better, negative => aggressive better)
#
# Problem 2:
#   [ ] run VI and PI for 100 iterations each
#   [ ] plot V after 10, 20, 50, 100 iterations (one subplot per method, or one per checkpoint)
#   [ ] convergence: plot max|V_k - V*| vs k for both (log y-axis); take V* from PI's final V
#   [ ] runtimes: time.perf_counter() around 100 iterations of each; if too fast, repeat with N=1000
#   [ ] plot V* - V_lazy and V* - V_aggr; check (diff >= 0).all() and where it's strictly > 0
#   [ ] plot / print the optimal policy: find the threshold state where it switches to a_high
#
# Save figures with plt.savefig("figs/<name>.png") so you can include them in Tex/RLProblemSet1.tex

import time
import numpy as np
import matplotlib.pyplot as plt

from QueueingEnvironment import QueueingMDP
from QueueingPolicies import (lazy_policy, aggressive_policy, policy_evaluation,
                             value_iteration, policy_iteration)


if __name__ == "__main__":
    # TODO
    pass
