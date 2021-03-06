import random
import numpy as np

episodes = 0
class Agent:
    EPS_START = 1.0
    EPS_STOP  = 0.3
    EPS_EPISODES = 8e3
    
    GAMMA = 0.99
    N_STEP_RETURN = 8
    GAMMA_N = GAMMA ** N_STEP_RETURN
    
    def __init__(self, brain, n_actions, start_epi=0, eps_start=EPS_START, eps_end=EPS_STOP, eps_episodes=EPS_EPISODES):
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_episodes = eps_episodes
        
        global episodes; episodes = start_epi
        
        self.brain = brain
        self.n_actions = n_actions
        
        self.memory = []    # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        "Calculate epsilon based on the current episode and the decay of epsilon"
        if(episodes >= self.eps_episodes):
            return self.eps_end
        else:
            # linearly interpolate
            return self.eps_start + episodes * (self.eps_end - self.eps_start) / self.eps_episodes    

    def act(self, s):
        """ Choose action
        
        With epsilon probability, take a random action
        Otherwise choose an action acording their probabilities
        """
        eps = self.getEpsilon()            

        if random.random() < eps:
            return random.randint(0, self.n_actions-1)

        else:
            s = np.array([s])
            p = self.brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(self.n_actions, p=p)

            return a
    
    def train(self, s, a, r, s_):
        """ Add the observation to the brain
        
        Calculate the accumulated reward over the last n-steps and add to the brain
        If the "next state" is None, i.e., the epsiode is over, push all the following observations to brain
        """
        def get_sample(memory, n):
            s, a, _, _  = memory[0]
            _, _, _, s_ = memory[n-1]

            return s, a, self.R, s_

        a_cats = np.zeros(self.n_actions)    # turn action into one-hot representation
        a_cats[a] = 1 

        self.memory.append( (s, a_cats, r, s_) )

        self.R = ( self.R + r * self.GAMMA_N ) / self.GAMMA

        if s_ is None:
            global episodes; episodes += 1
            
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