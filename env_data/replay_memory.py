import os
import sys
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as plt

class ReplayMemory():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.mem_length = FLAGS.mem_length
        self.count = 0
        pass

    def append(self, data_list):
        if self.count < self.mem_length:
            pass
        else:
            pass
        
        self.count += 1
