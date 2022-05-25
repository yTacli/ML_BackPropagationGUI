# Yücel TACLI

from math import sqrt
from random import random
import numpy as np
from Layer import Layer
from Noron import Noron

# hiddenNumber = []
def create_model_base(inputNumber,inputValue,hiddenLayerNumber,hiddenNumber,outputNumber,activatitionFunction='Sigmoid',threshold=0.0):
    model = []
    # input
    inputLayer = Layer()
    for i in range(inputNumber):
        noron = Noron(inputValue[i],activatitionFunction,threshold)        
        inputLayer.norons.append(noron)
    model.append(inputLayer)
    # hidden
    for i in range(hiddenLayerNumber):        
        hiddenLayer = Layer()  
        for j in range(hiddenNumber[i]):
            noron = Noron(0,activatitionFunction,threshold)  
            hiddenLayer.norons.append(noron)  
        model.append(hiddenLayer) 
    # output
    outputLayer = Layer() 
    for i in range(outputNumber):
        noron = Noron(0,activatitionFunction,threshold)
        outputLayer.norons.append(noron)
    model.append(outputLayer)    

    wights = random_weight(inputNumber,hiddenLayerNumber,hiddenNumber,outputNumber)
    biases = random_bias(hiddenLayerNumber,hiddenNumber,outputNumber)
    return model,wights,biases

def random_weight(inputNumber,hiddenLayerNumber,hiddenNumber,outputNumber):
    weights = []   
    inoronNumber = []   
    for i in range(inputNumber):
        nextNoron = []
        for j in range(hiddenNumber[0]): 
            nextNoron.append(random())
        inoronNumber.append(nextNoron)
    weights.append(inoronNumber)

    for i in range(hiddenLayerNumber-1):
        noronNumber = []
        for j in range(hiddenNumber[i]):           
            nextNoron = []
            for k in range(hiddenNumber[i+1]):               
                nextNoron.append(random())                
            noronNumber.append(nextNoron)
        weights.append(noronNumber)
    
    onoronNumber = [] 
    for i in range(hiddenNumber[hiddenLayerNumber-1]):   
        nextNoron = []    
        for j in range(outputNumber):
            nextNoron.append(random())
        onoronNumber.append(nextNoron)
    weights.append(onoronNumber)

    return weights

def random_bias(hiddenLayerNumber,hiddenNumber,outputNumber):
    biases = []    
    for i in range(hiddenLayerNumber):
        nextNoron = []
        for j in range(hiddenNumber[i]):
            nextNoron.append(random())
        biases.append(nextNoron)   

    nextNoron = []        
    for j in range(outputNumber):
        nextNoron.append(random())
    biases += [nextNoron]
    return biases

def forward(model,weight,bias): 
    md = model 
    for i in range(1,len(model)):    # layer       
        for j in range(len(model[i].norons)):   # noron
            sum = 0
            for k in range(len(model[i-1].norons)): # back noron
                sum += model[i-1].norons[k].value * weight[i-1][k][j]                              
            md[i].norons[j].value = sum + bias[i-1][j] 
    return md

def derivatives(activatitionFunction, x):
    if activatitionFunction == "Sigmoid":
        return x*(1-x)
    if activatitionFunction == "ReLU":
        if x >= 0:
            return 1
        else: 
            return 0

def error_coefficient_out(activatitionFunction,outputTarget,outputPredic):
    derivative = derivatives(activatitionFunction,outputPredic)   
    return (outputTarget-outputPredic) * derivative

# Hata Katsayısı 
def error_coefficient(models,weights,outputTarget,activatitionFunction):
    # Out     
    for i in range(len(models[len(models)-1].norons)):
        models[len(models)-1].norons[i].errorCoefficient = error_coefficient_out(outputTarget[i],models[len(models)-1].norons[i].value)    
    # Hidden
    for i in range(len(models)-2, 0, -1): # ilk katmana gerek yok
        for j in range(len(models[i].norons)):
            sum_error_weight = 0
            for k in range(len(models[i+1].norons)):
                sum_error_weight += models[i+1].norons[k].errorCoefficient * weights[i][j][k]
                derivative = derivatives(activatitionFunction,models[i].norons[j].value) 
            models[i].norons[j].errorCoefficient = sum_error_weight * derivative
    return models

# BACKWARD 
def update_weight_bias(models,weights,bias,learningRate):
    #weight
    updateWeights=[]
    for i in range(len(models)-1):
        noronNumber = []
        for j in range(len(models[i].norons)):
            nextNoron = []            
            for k in range(len(models[i+1].norons)):
                newWeight = weights[i][j][k] - learningRate * models[i+1].norons[k].errorCoefficient * models[i].norons[j].value               
                nextNoron.append(newWeight)                
            noronNumber.append(nextNoron)
        updateWeights.append(noronNumber)
    #bias
    updateBias = []
    for i in range(len(models)-1):
        nextNoron = []
        for j in range(len(models[i+1].norons)):
            newBias = bias[i][j] - learningRate * models[i+1].norons[j].errorCoefficient
            nextNoron.append(newBias)
        updateBias.append(nextNoron)   
    return updateWeights,updateBias

def backward(models,weights,bias,outputTarget,learningRate,activatitionFunction):    
    m = error_coefficient(models,weights,outputTarget,activatitionFunction)
    upW,upB = update_weight_bias(m,weights,bias,learningRate)    
    return m,upW,upB

# SSE 
def sum_square_error(models,outputTarget):
    outputPredic=[]
    for i in range(len(models[len(models)-1].norons)):
        outputPredic.append(models[len(models)-1].norons[i].value)
    error=0
    for i in range(len(outputTarget)):
        error += np.square(outputTarget[i]-outputPredic[i])     
    return error
# MSE 
def mean_square_error(models,outputTarget):
    outputPredic=[]
    for i in range(len(models[len(models)-1].norons)):
        outputPredic.append(models[len(models)-1].norons[i].value)
    error=0
    for i in range(len(outputTarget)):
        error += np.square(outputTarget[i]-outputPredic[i])     
    return error/len(outputTarget)
# RMSE
def root_mean_square_error(models,outputTarget):
    outputPredic=[]
    for i in range(len(models[len(models)-1].norons)):
        outputPredic.append(models[len(models)-1].norons[i].value)
    error=0
    for i in range(len(outputTarget)):
        error += np.square(outputTarget[i]-outputPredic[i])
    return sqrt(error/len(outputPredic))
