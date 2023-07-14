from typing import Tuple, List
import numpy as np


from environment import Environment
from agent import Agent


def simulate(
    n_trials: int, 
    n_agents: int, 
    structure: np.ndarray, 
    a_obs: np.ndarray, 
    b_obs: np.ndarray, 
    c_obs: np.ndarray, 
    emp_tau: float, 
    sticky_weight: float, 
    method: str='fullpost',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Simulates the environment with specified parameters.

    Args:
        n_trials (int): Number of trials.
        n_agents (int): Number of agents.
        structure (np.ndarray): Structure of the environment.
        a_obs, b_obs, c_obs (np.ndarray): Observations of agents A, B and C.
        emp_tau (float): Empirical tau parameter.
        sticky_weight (float): Sticky weight parameter.
        method (str): Method to be used, default is 'fullpost'.

    Returns:
        marginal_inferred_obs (Tuple[np.ndarray]): Marginal inferred observations for agents A, B, and C.
        environment_communications (np.ndarray): The communications in the environment.
    """

    env = Environment(n_trials, n_agents, structure=structure)
    A = Agent(agent_id=0, prior_a=1, prior_b=1, n_choices=7, 
              possible_evidence=[1, -1, 0], 
              evidence_probs=np.array([1/3, 1/3, 1/3]), 
              reliability=10, empirical_tau=emp_tau, 
              sticky_weight=sticky_weight, start_trial=0, 
              fixed_obs=a_obs) 

    B = Agent(agent_id=1, prior_a=1, prior_b=1, n_choices=7, 
              possible_evidence=[1, -1, 0], 
              evidence_probs=np.array([1/3, 1/3, 1/3]), 
              reliability=10, empirical_tau=emp_tau, 
              sticky_weight=sticky_weight, start_trial=0, 
              fixed_obs=b_obs)

    C = Agent(agent_id=2, prior_a=1, prior_b=1, n_choices=7, 
              possible_evidence=[1, -1, 0], 
              evidence_probs=np.array([1/3, 1/3, 1/3]), 
              reliability=10, empirical_tau=emp_tau, 
              sticky_weight=sticky_weight, start_trial=0, 
              fixed_obs=c_obs) 

    for _ in range(n_trials):
        Acom = A.step(env, method)
        Bcom = B.step(env, method)
        Ccom = C.step(env, method)

        env.update_env(trial=Acom[0], agent_id=Acom[1], observation=Acom[2], communication=Acom[3])
        env.update_env(trial=Bcom[0], agent_id=Bcom[1], observation=Bcom[2], communication=Bcom[3])
        env.update_env(trial=Ccom[0], agent_id=Ccom[1], observation=Ccom[2], communication=Ccom[3])

    return A.marginal_inferred_obs, B.marginal_inferred_obs, C.marginal_inferred_obs, env.communications