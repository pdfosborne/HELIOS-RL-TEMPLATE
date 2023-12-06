from datetime import datetime
import pandas as pd
# ====== HELIOS IMPORTS =========================================
# ------ Train/Test Function Imports ----------------------------
from helios_rl import STANDARD_RL
# ------ Environment Interaction Loop ---------------------------
from helios_rl.interaction_loops.standard import StandardInteractionLoop
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

if __name__=='__main__':
    main()