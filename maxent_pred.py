import sys
import os
import numpy as np
import scipy.special

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared_control_proj.shared_control import config

class MaxEntPredictor:
    def __init__(self, policies, clip=True, sensitivity=1):
        self._policies = policies
        self._log_probs = np.full((len(policies)), np.log(1./len(policies)))
        self._clip = clip
        self._sensitivity = sensitivity

    def flush_letter_probability(self, index):
        # Zero out the log probability for the given index
        self._log_probs[index] = -np.inf  # effectively setting the probability to 0

        # Re-normalize the log probabilities
        self._log_probs = self._log_probs - scipy.special.logsumexp(self._log_probs)

        # Clip the probabilities if required
        if self._clip:
            self._log_probs = config.clip_probability(self._log_probs)

    def get_prob(self):
        return np.exp(self._log_probs)
    
    def get_log_prob(self):
        return self._log_probs

    def update(self, x, u):
        q_vals = self._sensitivity * np.array([p.get_q_value(x, u) for p in self._policies])
        self._log_probs += q_vals - scipy.special.logsumexp(q_vals)
        self._log_probs = self._log_probs - scipy.special.logsumexp(self._log_probs)
        if self._clip:
            self._log_probs = config.clip_probability(self._log_probs)
        return self.get_prob()

    def get_prob_after_obs(self, x, u):
        q_vals = np.array([p.get_q_value(x, u) for p in self._policies])
        print(q_vals)
        log_probs = self._log_probs + q_vals - scipy.special.logsumexp(q_vals)
        log_probs = log_probs - scipy.special.logsumexp(log_probs)
        if self._clip:
            log_probs = config.clip_probability(log_probs)
        return np.exp(log_probs)





