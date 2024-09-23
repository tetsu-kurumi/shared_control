import sys
import os
import numpy as np
import scipy

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from shared_control_proj.shared_control import config

class ContinuousSharedAutoPolicy:
    def __init__(self, policies):
        self._policies = policies

    def get_base_actions(self, x):
        # for debugging/display
        return [ policy.get_q_grad(x) for policy in self._policies]

    def get_action(self, x, prob_policy, return_dist=False, sample=False):
        J = np.vstack([ policy.get_q_grad(x) for policy in self._policies ])
        print('used_pg:', prob_policy)
        action_weights = prob_policy @ J
        action = action_weights

        return action
    