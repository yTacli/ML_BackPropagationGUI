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
        elif self.activationFunction == 'ReLU':
            self.output = sigmoid(self.value)
        elif self.activationFunction == 'Threshold':
            self.output = threshold(self.value,self.threshold)                    

# Threshold (unit-step)
# Türevlenebilir olmadığı için error-coefficiation hatalı olacaktır.
# Unit-Step
def threshold(x,t): 
    if x >= t:
        return 1
    else:        
        return 0
        
# Sigmoid Aktivasyon
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    if x < 0:
        return 0
    elif x >= 0:
        return x

# Türevi Error-coefficiation'ı etkilediği için kullanılmadı
def tanh(x):
    # (e^x-e^-x)/(e^x+e^-x)
    return np.tanh(x)