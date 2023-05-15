from tqdm import tqdm
import time
# ------ Imports -----------------------------------------
from environment.engine import Engine
# Adapter
from adapters import default
# Agent Setup
from helios_rl.environment_setup.imports import ImportHelper
# Evaluation standards
from helios_rl.environment_setup.results_table import ResultsTable
from helios_rl.environment_setup.helios_info import HeliosInfo

STATE_ADAPTER_TYPES = {
    "Engine": default
}

class Environment:

    def __init__(self, local_setup_info: dict):
        # --- INIT env from engine
        engine = Engine()
        engine.generate()
        self.start_obs = engine.reset()
        self.legal_move_generator = engine.legal_move_generator()
        # ---

        # --- PRESET HELIOS INFO
        # Agent
        Imports = ImportHelper(local_setup_info)
        self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = Imports.agent_info(STATE_ADAPTER_TYPES)
        self.num_train_episodes, self.num_test_episodes, self.training_action_cap, self.testing_action_cap, self.reward_signal = Imports.parameter_info()  
        # Training or testing phase flag
        self.train = Imports.training_flag()
        # --- HELIOS
        self.live_env, self.observed_states, self.experience_sampling = Imports.live_env_flag()
        # Results formatting
        self.results = ResultsTable(local_setup_info)
        # HELIOS input function
        # - We only want to init trackers on first batch otherwise it resets knowledge
        self.helios = HeliosInfo(self.observed_states, self.experience_sampling)
        # Env start position for instr input
        # Enable sub-goals
        if (local_setup_info['sub_goal'] is not None) & (local_setup_info['sub_goal']!=["None"]) & (local_setup_info['sub_goal']!="None"):
            self.sub_goal:list = local_setup_info['sub_goal']
        else:
            self.sub_goal:list = None


    def episode_loop(self):
        # Mode selection (already initialized)
        if self.train:
            number_episodes = self.num_train_episodes
        else:
            number_episodes = self.num_test_episodes

        for episode in tqdm(range(0, number_episodes)):
            action_history = []
            # ---
            obs = self.start_obs
            legal_moves = self.legal_move_generator(obs)
            state = self.adapter.adapter(state=obs, legal_moves=legal_moves, episode_action_history=action_history, encode=True)
            # ---
            start_time = time.time()
            episode_reward:int = 0
            for action in range(0,self.training_action_cap):
                if self.live_env:
                    # Agent takes action
                    legal_moves = self.legal_move_generator(obs)
                    agent_action = self.agent.policy(state, legal_moves)
                    action_history.append(agent_action)
                    
                    next_obs, reward, terminated = self.env.step(state=obs, action=agent_action)
                    # Can override reward per action with small negative punishment
                    if reward==0:
                        reward = -0.05
                    
                    legal_moves = self.legal_move_generator(next_obs) 
                    next_state = self.adapter.adapter(state=next_obs, legal_moves=legal_moves, episode_action_history=action_history, encode=True)
                    # HELIOS trackers    
                    self.helios.observed_state_tracker(engine_observation=next_obs,
                                                        language_state=self.adapter.adapter(state=next_obs, legal_moves=self.legal_actions, episode_action_history=action_history, encode=False))
                    
                    # MUST COME BEFORE SUB-GOAL CHECK OR 'TERMINAL STATES' WILL BE FALSE
                    self.helios.experience_sampling_add(state, agent_action, next_state, reward, terminated)
                    # Trigger end on sub-goal if defined
                    if self.sub_goal:
                        if next_obs in self.sub_goal:
                            reward = self.reward_signal[0]
                            terminated = True                        
                else:
                    # Experience Sampling
                    legal_moves = self.helios.experience_sampling_legal_actions(state)
                    # Unknown state, have no experience to sample from so force break episode
                    if legal_moves == None:
                        break
                    
                    agent_action = self.agent.policy(state, legal_moves)
                    next_state, reward, terminated = self.helios.experience_sampling_step(state, agent_action)

                if self.train:
                    self.agent.learn(state, next_state, reward, agent_action)
                episode_reward+=reward
                if terminated:
                    break
                else:    
                    state=next_state
                    if self.live_env:
                        obs = next_obs
                        
            end_time = time.time()
            agent_results = self.agent.q_result()
            if self.live_env:
                self.results.results_per_episode(self.agent_name, None, episode, action, episode_reward, (end_time-start_time), action_history, agent_results[0], agent_results[1]) 

        return self.results.results_table_format()
                    