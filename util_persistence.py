import pickle
import os 

class Persister:
    
    def __init__(self, file_name):
        self.file_name = file_name
        
    def write(self, array):
        with open(self.file_name, 'wb+') as file:
            pickle.dump(array, file)
    
    def read(self):
        if not os.path.isfile(self.file_name):
            return []
        with open(self.file_name, 'rb') as file:
            return pickle.load(file)
