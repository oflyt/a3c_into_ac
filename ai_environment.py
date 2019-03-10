from util_image_preprocess import preprocess
from util_logger import Logger
import util_state as state_util
import util_display as display

import matplotlib.pyplot as plt

import numpy as np
import threading, time

logger = Logger(30)

class Environment(threading.Thread):
    stop_signal = False
    THREAD_DELAY = 0.001

    def __init__(self, env, rewards, agent, render=False):
        threading.Thread.__init__(self)

        self.render = render
        self.env = env
        self.agent = agent
        
        self.rewards = rewards
        
    def runEpisode(self):
        frame = self.env.reset()
        s = state_util.create(preprocess(frame))

        R = 0
        while True:         
            time.sleep(self.THREAD_DELAY) # yield 

            if self.render: 
                display.show_state(s, self.env.spec.id, 0, R)

            s1 = np.moveaxis(s, 0, -1)
            a = self.agent.act(s1)
            s_, r, done = self._step(self.env, a, s)

            if done: 
                s_1 = None
            else:
                s_1 = np.moveaxis(s_, 0, -1)
            
            self.agent.train(s1, a, r, s_1)

            s = s_
            R += r

            if done or self.stop_signal:
                self.rewards.append(R)
                break

        #logger.log("{}".format(R))

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True
        
    def _step(self, env, action, state):
        next_frame_1, reward_1, done_1, _ = env.step(action)
        next_frame_2, reward_2, done_2, _ = env.step(action)
        next_state = state_util.update(state, preprocess(next_frame_1), preprocess(next_frame_2))
        return (next_state, int(reward_1 + reward_2), done_1 or done_2)