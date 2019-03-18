import pickle
import os 

class Persister:
    
    def __init__(self, file_name):
        self.file_name = file_name
        
    def write(self, array):
        "Writed an array to file file_name"
        with open(self.file_name, 'wb+') as file:
            pickle.dump(array, file)
    
    def read(self):
        "Reads an array from file file_name, returns empty array if none exists"
        if not os.path.isfile(self.file_name):
            return []
        with open(self.file_name, 'rb') as file:
            return pickle.load(file)
