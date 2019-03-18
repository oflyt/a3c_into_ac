import threading, time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
from IPython import display 

class Plotter(threading.Thread):
    stop_signal = False

    def __init__(self, rewards, agent):
        threading.Thread.__init__(self)
        self.rewards = rewards
        self.agent = agent

    def run(self):
        "Main loop of thread"
        while not self.stop_signal:
            time.sleep(5)
            self._show(self.rewards)

    def stop(self):
        "Send stop signal to thread"
        self.stop_signal = True

    def _show(self, rewards):
        """Show the trend of the rewards gotten by the agents
        An average is calculated for every 200 rewards
        """
        step_size = 200
        
        size = len(rewards)
        n_points = size//step_size
        last_index = n_points*step_size
        
        y = np.array(rewards[0:last_index])
        
        x_smooth = range(0, last_index, step_size)
        y_smooth = np.mean(y.reshape(-1, step_size), axis=1)

        plt.plot(x_smooth, y_smooth)
        plt.ylabel('scores')
        plt.xlabel('episode')
        plt.title('Score over episodes')
        ax = plt.gca()
        ax.set_ylim(ymin=0)
        
        display.clear_output(wait=True)
        plt.show()
        
        print("Epsilon:", self.agent.getEpsilon())