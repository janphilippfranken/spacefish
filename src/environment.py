from typing import Dict, Tuple
import numpy as np


class Environment:
    
    def __init__(
    self, 
    n_trials: int, 
    n_agents: int, 
    structure: np.ndarray,
) -> None:
        """
        Initialize the environment with given trials, agents and structure.
        
        Args:
            n_trials (int): Number of trials.
            n_agents (int): Number of agents.
            structure (np.ndarray): Structure of the environment.
        """
        self.agents = n_agents
        self.trials = n_trials
        self.observations = np.zeros((self.trials, self.agents))
        self.communications = np.zeros((self.trials, self.agents))
        self.structure = structure
        
    def update_env(
        self, trial: int, 
        agent_id: int, 
        observation: int, 
        communication: int,
    ) -> None:
        """
        Update environment based on agent observation and communication.
        
        Args:
            trial (int): Trial number.
            agent_id (int): ID of the agent.
            observation (int): Observation from the agent.
            communication (int): Communication from the agent.
        """
        self.observations[trial, agent_id] = observation
        self.communications[trial, agent_id] = communication
        
    def get_env(
        self, 
        trial: int, 
        agent_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """
        Get available observations and communications from environment for agent.
        
        Args:
            trial (int): Trial number.
            agent_id (int): ID of the agent.

        Returns:
            observation (np.ndarray): Observations from the trials.
            communication (np.ndarray): Communications from other agents.
            parents (dict): Parent agents for the agent.
        """
        observation = self.observations[:trial + 1]
        communication = self.communications[:trial, np.where(self.structure[:,agent_id] == 1)[0]]
        
        parents = {}
        for agent in range(self.communications.shape[1]):
            parents[agent] =  np.where(self.structure[:,agent] == 1)[0]
        
        return observation, communication, parents