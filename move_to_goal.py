import numpy as np

def get_q_value(g, x, a, dist=5, flat_cost=1):
    d_to_goal = np.linalg.norm(g - x)
    dp_to_goal = np.linalg.norm(g - x - a)

    mult = 1
    if d_to_goal < dp_to_goal:
        mult = -1
        d_to_goal, dp_to_goal = dp_to_goal, d_to_goal

    if d_to_goal >= dist:
        if dp_to_goal >= dist:
            Q = (d_to_goal - dp_to_goal) * flat_cost
        else:
            Q = (d_to_goal - dist) * flat_cost +\
                0.5 * flat_cost * dist * (1 - (dp_to_goal/dist)**2)
    else:
        Q = 0.5 * flat_cost * (d_to_goal + dp_to_goal) * (d_to_goal - dp_to_goal) / dist
    return Q*mult

def get_q_grad(g, x, dist, flat_cost):
    v = g - x
    d = np.linalg.norm(v)
    if d >= dist:
        return flat_cost * v / d
    else:
        return 0.5 * flat_cost * v / d


class MoveToGoalPolicy:
    def __init__(self, goal, dist=5, flat_cost=1):
        self._goal = np.asanyarray(goal)
        self._dist = dist
        self._flat_cost = flat_cost

    def step(self, *args, **kwargs):
        pass

    def get_q_value(self, x, a):
        return get_q_value(self._goal, x, a, self._dist, self._flat_cost)
    
    def get_q_grad(self, x):
        return get_q_grad(self._goal, x, self._dist, self._flat_cost)

    @property
    def goal(self):
        return self._goal
