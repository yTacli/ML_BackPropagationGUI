# Yücel TACLI

import numpy as np
class Noron(): 
    # layer:int, value:float, inWeights:list, outWeights:list, activationFunction:str(S=sigmoid, T=threshold), t:int(threshold)
    def __init__(self, value,activationFunction='Sigmoid',threshold=0.0):
        self.value = value       
        self.errorCoefficient = 0
        self.activationFunction = activationFunction
        self.threshold = 0      
        if self.activationFunction == 'Sigmoid':
            self.output = sigmoid(self.value)
        elif self.activationFunction == 'Threshold':
            self.output = threshold(self.value,self.threshold)                    
    
# Threshold (unit-step)
def threshold(x,t):
    if x >= t:
        return 1
    else:
        return 0
        
# Sigmoid Aktivasyon
def sigmoid(x):
    return 1/(1+np.exp(-x))