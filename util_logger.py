from IPython import display 
from collections import deque
import numpy as np

class Logger:
    
    def __init__(self, n_values, header=None):
        self.n_values = int(n_values)
        self.log_queue = deque(maxlen=n_values*3)
        self.header = header
    
    def log(self, text):
        """ Log a new value to the output
        Will log the previous n_values*3 plus a header if provided.
        n_values values will be printed on one row
        """
        
        self.log_queue.append(text)
        display.clear_output(wait=True)
        if self.header != None:
            print(self.header)
         
        a = np.array(self.log_queue)
        b = a[0:self.n_rows] 
        c = a[self.n_rows:self.n_rows*2]
        d = a[self.n_rows*2:self.n_rows*3]

        print("\n".join([", ".join(b), ", ".join(c), ", ".join(d)]), flush=True)
