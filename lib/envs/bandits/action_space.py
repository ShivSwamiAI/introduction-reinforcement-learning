class ActionSpace(object):
    
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)