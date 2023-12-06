from datetime import datetime
import pandas as pd
# ====== HELIOS IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from helios_rl import STANDARD_RL
# ------ Environment Interaction Loop ---------------------------
from helios_rl.interaction_loops.standard import StandardInteractionLoop
# ------ HELIOS Hierarchies -------------------------------------
from helios_rl import HELIOS_SEARCH
from helios_rl import HELIOS_OPTIMIZE
# ------ Config Import ------------------------------------------
# Meta parameters
from helios_rl.config import TestingSetupConfig
# Local parameters
from helios_rl.config_local import ConfigSetup
# ====== LOCAL IMPORTS ==========================================
# ------ Local Data MDP Engine ----------------------------------
from environment.engine import Engine
# ------ Local Adapters -----------------------------------------
from adapters.default import DefaultAdapter
from adapters.language import LanguageAdapter
# ------ Combined Analysis --------------------------------------
from helios_rl import COMBINED_VARIANCE_ANALYSIS_GRAPH

ADAPTERS = {
    'Default': DefaultAdapter,
    'Language': LanguageAdapter
}

def main():
    # ------ Load Configs -----------------------------------------
    # Meta parameters
    ExperimentConfig = TestingSetupConfig("./config.json").state_configs
    # Local Parameters
    ProblemConfig = ConfigSetup("./config_local.json").state_configs

    Environment = StandardInteractionLoop(Engine, ADAPTERS, ProblemConfig)

    # Specify save dir
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    save_dir = './output/'+str('test')+'_'+time 

    # --------------------------------------------------------------------
    # Flat Baselines
    flat = STANDARD_RL(Config=ExperimentConfig, Environment=Environment,
                save_dir=save_dir, show_figures = 'No', window_size=0.1)
    flat.train()  
    flat.test()
    # --------------------------------------------------------------------
    # HELIOS Instruction Following
    num_plans = 1
    num_explor_epi = 20
    sim_threshold = 0.95

    observed_states = None
    instruction_results = None
    
    helios = HELIOS_SEARCH(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                        Environment=Environment,
                        save_dir = save_dir+'/Reinforced_Instr_Experiment',
                        num_plans = num_plans, number_exploration_episodes=num_explor_epi, sim_threshold=sim_threshold,
                        feedback_increment = 0.25, feedback_repeats=1,
                        observed_states=observed_states, instruction_results=instruction_results)

    # Don't provide any instruction information, will be defined by command line input
    helios_results = helios.search(action_cap=10, re_search_override=False, simulated_instr_goal=None)

    # Store info for next plan -> assumes we wont see the same instruction twice in one plan
    observed_states = helios_results[0]
    instruction_results = helios_results[1]
    # Take Instruction path now defined with reinforced+unsupervised sub-goal locations and train to these
    # Init experiment setup with sub-goal defined
    reinforced_experiment = HELIOS_OPTIMIZE(Config=ExperimentConfig, LocalConfig=ProblemConfig, 
                    Environment=Environment,
                    save_dir=save_dir+'/Reinforced_Instr_Experiment', show_figures = 'No', window_size=0.1,
                    instruction_path=None, predicted_path=instruction_results, instruction_episode_ratio=0.1)
    reinforced_experiment.train()
    reinforced_experiment.test()
    # --------------------------------------------------------------------
    # Combined Analysis
    flat_results = pd.read_csv(save_dir+'/Standard_Experiment'+'/testing_variance_results.csv')
    reinforced_results = pd.read_csv(save_dir+'/Reinforced_Instr_Experiment'+'/testing_variance_results.csv')

    variance_results = {}
    variance_results['Flat_agent'] = {}
    variance_results['Flat_agent']['results'] = flat_results
    variance_results['Flat_agent']['env_name'] = flat_results['agent'].iloc[0]
    variance_results['Flat_agent']['num_repeats'] = flat_results['num_repeats'].iloc[0]

    variance_results['Reinforced_instructions'] = {}
    variance_results['Reinforced_instructions']['results'] = reinforced_results
    variance_results['Reinforced_instructions']['env_name'] = reinforced_results['agent'].iloc[0]
    variance_results['Reinforced_instructions']['num_repeats'] = reinforced_results['num_repeats'].iloc[0]

    COMBINED_VARIANCE_ANALYSIS_GRAPH(variance_results, save_dir, show_figures='N')
    # --------------------------------------------------------------------

if __name__=='__main__':
    main()