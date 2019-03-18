import matplotlib.pyplot as plt
from IPython import display
import numpy as np

def show_state(observation, env_id, step=0, info=""):
    """ Disply the current state
    
    Before anything is displayed, all previous output is cleared.
    
    If 4 frames are sent in, i.e., a state. 
    Then the frames are displyed on top of each other 
    with newer frames having higher brightness. 
    
    Overwise, 1 frame is expected and it will be displayed directly, as is. 
    """
    plt.figure(3)
    plt.clf()
    
    if len(observation)==4:
        frames = list()
        for i, val in enumerate([0.1, 0.2, 0.3, 0.4]):
            frames.append(np.multiply(observation[i], val))
        observation = np.sum(frames, axis=0)
        
    plt.imshow(observation, cmap='gray')
    plt.axis('off')
    plt.title("%s | Score: %s" % (env_id, info))
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.close()