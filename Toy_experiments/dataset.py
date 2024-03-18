import numpy as np
from matplotlib import pyplot as plt



class GaussianDataset():
    def __init__(self, maj_size=5000, min_size=160, std=0.3):
        
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
        
        
class GaussianDataset3D():
    def __init__(self, seed=None, maj_size=3000, min_size=1000, std=0.25):
        
        self.std_maj = std
        self.std_min = std * 0.7
        self.maj_size = maj_size
        self.min_size = min_size
        self.grouped_embs = {}
        
        if seed is not None:
            self.set_seed(seed)
        
        self.core_ax, self.sp_ax, self.perp_ax = self.generate_random_axes()
        self.generate_dataset()

    def set_seed(self, seed):
        np.random.seed(seed)
        
    def normalize(self, x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def generate_random_axes(self):
        core_axis = self.normalize(2 * np.random.rand(3) - 1)
        sp_axis = self.normalize(2 * np.random.rand(3) - 1)
        dummy_axis = self.normalize(2 * np.random.rand(3) - 1)
        perpendicular_axis = self.normalize(np.cross(core_axis, dummy_axis))
        return core_axis, sp_axis, perpendicular_axis
    
    def generate_dataset(self, alpha=0.5):
        
        mean = -alpha * self.sp_ax + (1 - alpha) * self.core_ax
        min0 = np.random.normal(mean, self.std_min, size=(self.min_size, 3))
        min0 = self.normalize(min0)
        
        mean = (1 - alpha) * self.sp_ax + alpha * self.core_ax
        maj0 = np.random.normal(mean, self.std_maj, size=(self.maj_size, 3))
        maj0 = self.normalize(maj0)

        mean = alpha * self.sp_ax - (1 - alpha) * self.core_ax
        min1 = np.random.normal(mean, self.std_min, size=(self.min_size, 3))
        min1 = self.normalize(min1)
        
        mean = -(1 - alpha) * self.sp_ax - alpha * self.core_ax
        maj1 = np.random.normal(mean, self.std_maj, size=(self.maj_size, 3))
        maj1 = self.normalize(maj1)
        
        ood0 = np.random.normal(self.perp_ax, self.std_maj, size=(self.maj_size, 3))
        ood0 = self.normalize(ood0)
        
        
        self.grouped_embs['0_0'] = maj0
        self.grouped_embs['0_1'] = min0

        self.grouped_embs['1_0'] = min1
        self.grouped_embs['1_1'] = maj1
        
        data = np.concatenate([min0, maj0, min1, maj1], 0)
        labels = np.concatenate([np.zeros(self.min_size + self.maj_size), np.ones(self.min_size + self.maj_size)])
        group_labels = np.concatenate([0 * np.ones(self.min_size), 1 * np.ones(self.maj_size), 2 * np.ones(self.min_size), 3 * np.ones(self.maj_size)])
        
        permutation = np.random.permutation(2 * (self.min_size + self.maj_size))

        self.x = data[permutation]
        self.y = labels[permutation]
        self.g = group_labels[permutation]
        self.o = [ood0]
        

