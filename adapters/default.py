from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
# StateAdapter includes static methods for adapters
from helios_rl.encoders.poss_state_encoded import StateEncoder

class DefaultAdapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        # TODO: Update this based on the current problem, each requires preset knowledge of all possible states/actions/objects
        # - Possible Atates
        # - Possible Actions
        # - Prior Actions
        # - Possible Objects
    
        # Initialise encoder based on all possilbe env states
        all_possible_states = [i for i in range(4*4)]
        self.encoder = StateEncoder(all_possible_states)
        # --------------------------------------------------------------------
        # ONLY IF USING GYMNASIUM BASED AGENTS
        # - Observation space is required for Gym based agent, prebuilt HELIOS encoders provide this (TODO)
        # self.observation_space = self.encoder.observation_space
        # - Otherwise defined here:
        #   - See https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
        #   - Observations are dictionaries with the agent's and the target's location.
        #   - Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )
    
    def adapter(self, state:any, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
       
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in DefaultAdapter._cached_state_idx):
                    DefaultAdapter._cached_state_idx[sent] = len(DefaultAdapter._cached_state_idx)
                state_indexed.append(DefaultAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded