# Yücel TACLI
import random
import numpy as np
from Layer import Layer
from Noron import Noron

def random_weight(numInput,hiddenLayersNorons,numOutput,seed):
    weights = []      
    i2h = np.zeros(shape=[numInput,hiddenLayersNorons[0]], dtype=np.float64) # input two hidden
    weights.append(i2h)
    for i in range(len(hiddenLayersNorons)-1):
        h2h = np.zeros(shape=[hiddenLayersNorons[i],hiddenLayersNorons[i+1]], dtype=np.float64) # hidden two hidden
        weights.append(h2h)
    h2o = np.zeros(shape=[hiddenLayersNorons[-1],numOutput], dtype=np.float64) # hidden two output
    weights.append(h2o)     
    prew_weights_delta = weights
    lo = -0.01; hi = 0.01 
    rnd = random.Random(seed)
    for idx in weights:
        ix,iy=idx.shape
        for i in range(ix):
            for j in range(iy):
                idx[i,j] = (hi-lo) * rnd.random() + lo    
    return weights, prew_weights_delta
   
def random_bias(hiddenLayersNorons,numOutput,seed):    
    biases = []        
    for i in range(len(hiddenLayersNorons)):      
        h2h = np.zeros(shape=[hiddenLayersNorons[i]],dtype=np.float64)   # hidden two hidden           
        biases.append(h2h)
    h2o = np.zeros(shape=[numOutput], dtype=np.float64)     # hidden two output        
    biases.append(h2o)
    prew_bias_delta = biases
    rnd = random.Random(seed)
    lo = -0.01; hi = 0.01
    for bdx in biases:       
        for i in range(len(bdx)):             
            bdx[i] = (hi-lo) * rnd.random() + lo + lo  
    return biases, prew_bias_delta

# hiddenNumber = []
def create_model_base(numInput,inputValues,numHiddenLayer,hiddenLayersNorons,numOutput,seed,activatitionFunction='Sigmoid',threshold=0.0):           
    model = [] 
    # input
    inputLayer = Layer()
    for i in range(numInput):
        noron = Noron(inputValues[i],activatitionFunction,threshold)        
        inputLayer.norons.append(noron)
    model.append(inputLayer)
    # hidden
    for i in range(numHiddenLayer):        
        hiddenLayer = Layer()  
        for j in range(hiddenLayersNorons[i]):
            noron = Noron(0,activatitionFunction,threshold)  
            hiddenLayer.norons.append(noron)  
        model.append(hiddenLayer) 
    # output
    outputLayer = Layer() 
    for i in range(numOutput):
        noron = Noron(0,activatitionFunction,threshold)
        outputLayer.norons.append(noron)
    model.append(outputLayer)    
    
    weights,prew_weights_delta = random_weight(numInput,hiddenLayersNorons,numOutput,seed)
    biases,prew_bias_delta = random_bias(hiddenLayersNorons,numOutput,seed)
    return model,weights,biases,prew_weights_delta,prew_bias_delta

def forward(model,weights,biases):     
    for i in range(1,len(model)):    # layer       
        for j in range(len(model[i].norons)):   # noron
            sum = 0.0
            for k in range(len(model[i-1].norons)): # back noron                             
                sum += model[i-1].norons[k].value * weights[i-1][k,j]
            model[i].norons[j].value = sum + biases[i-1][j]             
    return model

def error_coefficient_out(outputTarget,outputPredic):
    return (outputTarget - outputPredic) * (outputPredic*(1-outputPredic))

# Hata Katsayısı 
def error_coefficient(models,weights,outputTarget):
    # Out     
    for i in range(len(models[len(models)-1].norons)):
        models[len(models)-1].norons[i].errorCoefficient = error_coefficient_out(outputTarget[i],models[len(models)-1].norons[i].value)    
    # Hidden
    for i in range(len(models)-2, 0, -1): # ilk katmana gerek yok
        for j in range(len(models[i].norons)):
            sum_error_weight = 0
            for k in range(len(models[i+1].norons)):
                sum_error_weight += models[i+1].norons[k].errorCoefficient * weights[i][j,k]                
            models[i].norons[j].errorCoefficient = sum_error_weight * (models[i].norons[j].value * (1 - models[i].norons[j].value))
    return models

# BACKWARD 
def update_weight_bias(models,weights,bias,learningRate,momentum,prewWeightsDelta,prewBiasDelta):
    updateWeights = weights
    #weight
    for i in range(len(weights)):
        ix,iy = weights[i].shape
        for j in range(ix):
            for k in range(iy):
                delta = learningRate * models[i+1].norons[k].errorCoefficient  *  models[i].norons[j].value       
                newWeight = weights[i][j,k] + delta
                newWeight += momentum * prewWeightsDelta[i][j,k]
                updateWeights[i][j,k] = newWeight
                prewWeightsDelta[i][j,k] = delta   
    #bias
    updateBias = bias
    for i in range(len(bias)): 
        for j in range(len(bias[i])):
            delta = learningRate * models[i+1].norons[j].errorCoefficient  *  1.0
            newBias = bias[i][j] + delta            
            newBias += momentum * prewBiasDelta[i][j]                        
            updateBias[i][j] = newBias            
            prewBiasDelta[i][j] = delta 
    return updateWeights,updateBias,prewWeightsDelta,prewBiasDelta

def backward(models,weights,bias,outputTarget,learningRate,momentum,prewWeightsDelta,prewBiasDelta):    
    netWork = error_coefficient(models,weights,outputTarget)
    upW,upB,prewW,prewB = update_weight_bias(netWork,weights,bias,learningRate,momentum,prewWeightsDelta,prewBiasDelta)  
    return netWork,upW,upB,prewW,prewB

def accuracy(models,weights,bias,selectRowInput,selectRowOutput):  # train or test data matrix
    numCorrect = 0; numWrong = 0
    for i in (len(models[0].norons)):
        models[0].norons[i].value = selectRowInput[i]

    y_values = forward(models,weights,bias)  # computed output values)
    max_index = np.argmax(y_values)  # index of largest output value 

    if abs(selectRowOutput[max_index] - 1.0) < 1.0e-5:
        numCorrect += 1
    else:
        numWrong += 1
    return (numCorrect * 1.0) / (numCorrect + numWrong)

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
    return np.sqrt(error/len(outputPredic))
