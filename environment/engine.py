
class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self) -> None:
        """Initialize Engine"""
        self.Environment = "Engine Initialization"
    
    def ledger(self) -> None:
        """Ledger of the environment with meta information for the problem"""
        self.required = {
            'id': 'Unique Problem ID',
            'description': 'Problem Description',
            'goal': 'Goal Description',
            }
        
        self.optional = {
            'name': 'Problem Name',
            'reward': 'Reward Description',
            'punishment': 'Punishment Description (if any)',
            'state': 'State Description',
            'constraints': 'Constraints Description',
            'action': 'Action Description',
            'author': 'Author',
            'year': 'Year'
        }

    def reset(self):
        """Fully reset the environment."""
        obs, _ = self.Environment.reset()
        return obs

    
    def step(self, state:any, action:any):
        """Enact an action."""
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="ENV_RESET"):
            self.reset()
            
        obs, reward, terminated = self.Environment.step(action)
        return obs, reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = self.Environment.legal_moves(obs)
        return legal_moves
    
    def render(self, state:any):
        """Render the environment."""
        render = self.Environment.render(state)
        return render

