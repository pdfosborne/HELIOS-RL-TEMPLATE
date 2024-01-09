
class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self, local_setup_info:dict={}) -> None:
        """Initialize Engine"""
        self.Environment = "Engine Initialization"
        self.setup_info = local_setup_info
    
    def ledger(self) -> None:
        """Ledger of the environment with meta information for the problem"""
        self.ledger_required = {
            'id': 'Unique Problem ID',
            'type': 'Language/Numeric',
            'description': 'Problem Description',
            'goal': 'Goal Description'
            }
        
        self.ledger_optional = {
            'reward': 'Reward Description',
            'punishment': 'Punishment Description (if any)',
            'state': 'State Description',
            'constraints': 'Constraints Description',
            'action': 'Action Description',
            'author': 'Author',
            'year': 'Year',
            'render_data':{'render_mode':'rgb_array', 
                           'render_fps':4}
        }
        self.ledger_gym_compatibility = {
            # Limited to discrete actions for now, set to arbitrary large number if uncertain
            'action_space_size':4, 
        }

    def reset(self):
        """Fully reset the environment."""
        obs, _ = self.Environment.reset()
        self.action_history = []
        return obs
    
    def step(self, state:any, action:any):
        """Enact an action."""
        # Record action history
        self.action_history.append(action)
        # Action space will always be numeric, step function may need to #
        # convert to form to match underlying engine, example:
        self._action_to_outcome = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left'
        }
        # Return outcome of action
        obs, reward, terminated, info = self.Environment.step(action)
        return obs, reward, terminated, info

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = self.Environment.legal_moves(obs)
        return legal_moves
    
    def render(self, state:any):
        """Render the environment."""
        render = self.Environment.render(state, self.optional['render_data'])
        return render
    
    def close(self):
        """Close/Exit the environment."""
        self.Environment.close()

