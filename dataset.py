import numpy as np
from matplotlib import pyplot as plt



class GaussianDataset():
    def __init__(self, maj_size=5000, min_size=200, std=0.3):
        
        self.std_maj = std
        self.std_min = std / 2
        self.maj_size = maj_size
        self.min_size = min_size
        
        self.d_core = 0.4
        self.d_sp = 1.
        
        self.generate_dataset()
        
    
    def generate_dataset(self):
        
        min0 = np.random.normal([-self.d_core, -self.d_sp], self.std_min, size=(self.min_size, 2))
        maj0 = np.random.normal([-self.d_core, self.d_sp], self.std_maj, size=(self.maj_size, 2))

        min1 = np.random.normal([self.d_core, self.d_sp], self.std_min, size=(self.min_size, 2))
        maj1 = np.random.normal([self.d_core, -self.d_sp], self.std_maj, size=(self.maj_size, 2))
        
        ood0 = np.random.normal([0, 3*self.d_sp], self.std_maj, size=(self.maj_size, 2))
        ood1 = np.random.normal([0, -3*self.d_sp], self.std_maj, size=(self.maj_size, 2))

        
        data = np.concatenate([min0, maj0, min1, maj1], 0)
        labels = np.concatenate([np.zeros(self.min_size + self.maj_size), np.ones(self.min_size + self.maj_size)])
        group_labels = np.concatenate([0 * np.ones(self.min_size), 1 * np.ones(self.maj_size), 2 * np.ones(self.min_size), 3 * np.ones(self.maj_size)])
        
        permutation = np.random.permutation(2 * (self.min_size + self.maj_size))

        self.x = data[permutation]
        self.y = labels[permutation]
        self.g = group_labels[permutation]
        self.o = [ood0, ood1]
        
        

