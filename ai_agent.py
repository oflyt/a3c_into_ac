import random
import numpy as np

frames = 0
class Agent:
    EPS_START = 1.0
    EPS_STOP  = 0.3
    EPS_STEPS = 75000
    
    GAMMA = 0.99
    N_STEP_RETURN = 8
    GAMMA_N = GAMMA ** N_STEP_RETURN
    
    def __init__(self, brain, n_actions, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_steps = eps_steps
        
        self.brain = brain
        self.n_actions = n_actions
        
        self.memory = []    # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if(frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps    # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()            
        global frames; frames = frames + 1

        if random.random() < eps:
            return random.randint(0, self.n_actions-1)

        else:
            s = np.array([s])
            p = self.brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(self.n_actions, p=p)

            return a
    
    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _  = memory[0]
            _, _, _, s_ = memory[n-1]

            return s, a, self.R, s_

        a_cats = np.zeros(self.n_actions)    # turn action into one-hot representation
        a_cats[a] = 1 

        self.memory.append( (s, a_cats, r, s_) )

        self.R = ( self.R + r * self.GAMMA_N ) / self.GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)

                self.R = ( self.R - self.memory[0][2] ) / self.GAMMA
                self.memory.pop(0)        

            self.R = 0

        if len(self.memory) >= self.N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, self.N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)    
    
    # possible edge case - if an episode ends in <N steps, the computation is incorrect