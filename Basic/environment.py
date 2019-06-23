from vizdoom import *

class Environment:
    def __init__(self, config_path, scenario_path):
        self.game = DoomGame()
        
        self.game.load_config(config_path)
        self.game.set_doom_scenario_path(scenario_path)
        
        self.game.init()

    def new_episode(self):
        self.game.new_episode()

    def make_action(self, action):
        reward = self.game.make_action(action)
        return reward
    def is_episode_finished(self):
        done = self.game.is_episode_finished()
        return done
        
    def get_state(self):
        state = self.game.get_state().screen_buffer
        return state

    def get_available_buttons_size(self):
        action_size = self.game.get_available_buttons_size()
        return action_size