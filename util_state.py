import numpy as np
from collections import deque

def update(state, frame_1, frame_2):
    "Update the array to have frame_1 and frame_2 at index 2 and 3 respectively"
    next_state = state.copy()
    next_state.append(frame_1)
    next_state.append(frame_2)
    return next_state

def create(frame):
    "Create an array of 4 duplicates of the same frame"
    state = deque(maxlen=4)
    for _ in range(4):
        state.append(frame)
    return state

if __name__ == '__main__':
    ## Simple testing of the functions
    frame1 = np.array([1])
    frame2 = np.array([2])
    frame3 = np.array([3])
    expected = np.array([frame1, frame1, frame2, frame3])
    
    state = create_state_arr(frame1)
    next_state = update_state_arr(state, frame2, frame3)
    
    if (next_state==expected).all():
        print("Next state as expected")
    else:
        print("Next state is wrong")
        
    for i in range(4, 16):
        state = next_state
        next_state = update_state_arr(state, np.array([i]), np.array([i+1]))
        for i in range(4):
            if (next_state[i]==state[i]).all():
                print("States are equal.. {}={} for index {}".format(next_state[i], expected[i], i))
    
    print("Done with tests")