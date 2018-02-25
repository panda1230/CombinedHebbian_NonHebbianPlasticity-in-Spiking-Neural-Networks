'''
Paper Title: Learning to Generate Sequences with Combination of Hebbian and Non-hebbian Plasticity in Recurrent Spiking Neural Networks
https://www.frontiersin.org/articles/10.3389/fnins.2017.00693/full
@author: Priyadarshini Panda and Kaushik Roy

This code randomly initializes the weights of the spiking reservoir.
'''

import scipy.ndimage as sp
import numpy as np
import pylab
import os
import sys
import getopt

stoc_enable = 0
def randomDelay(minDelay, maxDelay):
    return np.random.rand()*(maxDelay-minDelay) + minDelay

def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0]*int(numTargetWeights)
    while numWeights < numTargetWeights:
        idx = (np.int32(np.random.rand()*baseMatrix.shape[0]), np.int32(np.random.rand()*baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList

def create_weights():
    # LSM-SNN topology: [INPUT(784) X LIQUID(400) X OUTPUT(100x100)]
    nInput     = 784                         # Number of input neurons
    nLiquid    = 200                        # Number of liquid neurons
    nLiquidE   = int(np.ceil(0.80*nLiquid))  # Number of liquid-excitatory neurons
    nLiquidI   = nLiquid - nLiquidE          # Number of liquid-inhibitory neurons
    nE         = 60                         # Number of output-excitatory neurons
    nI         = nE                          # Number of output-inhibitory neurons
    
    # Initialize a seed for the random number generators
    np.random.seed(0)

    # Configure the output directory
    if(stoc_enable == 0):
       dataPath    = './random/'
       outDataPath = './weights/'
    else:
       dataPath    = './random_stoc/'
       outDataPath = './weights_stoc/'

    # Program the connectivity from input to liquid layer
    pLiquid = {}
    pLiquid['ee_input'] = 0.4  # Input to liquid-excitatory
    pLiquid['ei_input'] = 0.1  # 0.0 Input to liquid-inhibitory (NO CONNECTIONS)

    weightLiquid = {}
    weightLiquid['ee_input'] = 0.3
    weightLiquid['ei_input'] = 0.2 #0.0

    # Create random connections between input(Xe) and liquid-excitatory(Le) neurons
    print 'Create a random connection matrix between input(Xe) and liquid-excitatory(Le) neurons'
    connNameList = ['XeLe']
    if(stoc_enable == 0):
       for name in connNameList:
           weightMatrix  = np.random.random((nInput, nLiquidE)) + 0.01
           weightMatrix *= weightLiquid['ee_input']
           #print weightMatrix
           if(pLiquid['ee_input'] < 1):
              weightMatrix, weightList = sparsenMatrix(weightMatrix, pLiquid['ee_input'])
           else:
              weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidE) for i in xrange(nInput)]
           #print np.size(weightMatrix)
           #print np.count_nonzero(weightMatrix)
           #print weightMatrix
           #print 'Save connection matrix', dataPath+name, '\n'
           np.save(dataPath+name, weightList)
    else:
       for name in connNameList:
           weightMatrix  = (np.random.random((nInput, nLiquidE)) < pLiquid['ee_input']) * 1.
           weightMatrix += (1-weightMatrix)*weightLiquid['ee_input']
           weightList    = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidE) for i in xrange(nInput)]
           
           print 'Save connection matrix', dataPath+name, '\n'
           np.save(dataPath+name, weightList)

    # Create random connections between input(Xe) and liquid-inhibitory(Li) neurons
    print 'Create a random connection matrix between input(Xe) and liquid-inhibitory(Li) neurons'
    connNameList = ['XeLi']
    if(stoc_enable == 0):
       for name in connNameList:
           weightMatrix  = np.random.random((nInput, nLiquidI)) + 0.01
           weightMatrix *= weightLiquid['ei_input']
           if(pLiquid['ei_input'] < 1):
              weightMatrix, weightList = sparsenMatrix(weightMatrix, pLiquid['ei_input'])
           else:
              weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidI) for i in xrange(nInput)]

           # Assert that there are no connections from input to liquid-inhibitory neurons
           #assert(np.size(np.where(weightMatrix==0)[0]) == (nInput*nLiquidI))

           print 'Save connection matrix', dataPath+name, '\n'
           np.save(dataPath+name, weightList)
    else:
       for name in connNameList:
           weightMatrix  = (np.random.random((nInput, nLiquidI)) < pLiquid['ei_input']) * 1.
           weightMatrix += (1-weightMatrix)*weightLiquid['ei_input']
           weightList    = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidI) for i in xrange(nInput)]

           # Assert that there are no connections from input to liquid-inhibitory neurons
           assert(np.size(np.where(weightMatrix==0)[0]) == (nInput*nLiquidI))

           print 'Save connection matrix', dataPath+name, '\n'
           np.save(dataPath+name, weightList)

    # Program recurrent connectivity within the liquid layer
    connLambda = 2.
    connLiquid = {}
    connLiquid['ee'] = 0.3  # Liquid-excitatory to liquid-excitatory (0.3)
    connLiquid['ei'] = 0.3  # Liquid-excitatory to liquid-inhibitory (0.2)
    connLiquid['ie'] = 0.3  # Liquid-inhibitory to liquid-excitatory (0.4)
    connLiquid['ii'] = 0.3  # Liquid-inhibitory to liquid-inhibitory (0.1)

    weightLiquid['ee'] = 0.3 #1.0   # Default: 1.2
    weightLiquid['ei'] = 1.0   # Default: 1.6
    weightLiquid['ie'] = 1.0   # Default: 3.0
    weightLiquid['ii'] = 1.0   # Default: 2.8

    # Create random recurrent connections among the liquid-excitatory (Le-Le) neurons
    print 'Create a random connection matrix among the liquid-excitatory(Le-Le) neurons'
    connNameList = ['LeLe']
    for name in connNameList:
        weightMatrix  = np.random.random((nLiquidE, nLiquidE)) + 0.01
        weightMatrix *= weightLiquid['ee']
        print weightMatrix
        if(connLiquid['ee'] < 1):
            weightMatrix, weightList = sparsenMatrix(weightMatrix, connLiquid['ee'])
        else:
            weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidE) for i in xrange(nLiquidE)]
           #print np.size(weightMatrix)
           #print np.count_nonzero(weightMatrix)
        b=np.where(weightMatrix>0)
        print 'Percentage of active Le-Le connections =', (np.size(b[0])*100./(nLiquidE*nLiquidE)), '%'
        #print weightMatrix
        print 'Save connection matrix', dataPath+name, '\n'
        np.save(dataPath+name, weightList)
##    for name in connNameList:
##        weightMatrix  = ((np.random.random((nLiquidE, nLiquidE)) < connLiquid['ee']) * 1.)
##        print weightMatrix
##        print 'Percentage of active Le-Le connections =', (np.sum(weightMatrix)*100.)/(nLiquidE*nLiquidE), '%'
##        weightMatrix *= weightLiquid['ee']
##        weightList    = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidE) for i in xrange(nLiquidE)]
##
##        print 'Save connection matrix', dataPath+name, '\n'
##        np.save(dataPath+name, weightList)

    # Create random recurrent connections between liquid-excitatory(Le) and liquid-inhibitory(Li) neurons
    print 'Create a random recurrent connection matrix between liquid-excitatory(Le) and liquid-inhibitory(Li) neurons'
    connNameList = ['LeLi']
    for name in connNameList:
        weightMatrix  = ((np.random.random((nLiquidE, nLiquidI)) < connLiquid['ei']) * 1.)
        print 'Percentage of active Le-Li connections =', (np.sum(weightMatrix)*100.)/(nLiquidE*nLiquidI), '%'
        weightMatrix *= weightLiquid['ei']
        weightList    = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidI) for i in xrange(nLiquidE)]

        print 'Save connection matrix', dataPath+name, '\n'
        np.save(dataPath+name, weightList)

    # Create random recurrent connections between liquid-inhibitory(Li) and liquid-excitatory(Le) neurons
    print 'Create a random recurrent connection matrix between liquid-inhibitory(Li) and liquid-excitatory(Le) neurons'
    connNameList = ['LiLe']
    for name in connNameList:
        weightMatrix  = ((np.random.random((nLiquidI, nLiquidE)) < connLiquid['ie']) * 1.)
        print 'Percentage of active Li-Le connections =', (np.sum(weightMatrix)*100.)/(nLiquidI*nLiquidE), '%'
        weightMatrix *= weightLiquid['ie']
        weightList    = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidE) for i in xrange(nLiquidI)]

        print 'Save connection matrix', dataPath+name, '\n'
        np.save(dataPath+name, weightList)

    # Create random recurrent connections among the liquid-inhibitory (Li-Li) neurons
    print 'Create a random connection matrix among the liquid-inhibitory(Li-Li) neurons'
    connNameList = ['LiLi']
    for name in connNameList:
        weightMatrix  = ((np.random.random((nLiquidI, nLiquidI)) < connLiquid['ii']) * 1.)
        print 'Percentage of active Li-Li connections =', (np.sum(weightMatrix)*100.)/(nLiquidI*nLiquidI), '%'
        weightMatrix *= weightLiquid['ii']
        weightList    = [(i, j, weightMatrix[i,j]) for j in xrange(nLiquidI) for i in xrange(nLiquidI)]

        print 'Save connection matrix', dataPath+name, '\n'
        np.save(dataPath+name, weightList)

    # Program the connectivity from liquid to output layer
    # The liquid state is fed as an input to the output-excitatory neurons
    pConn = {}
    pConn['ee_input'] = 1.0   # Liquid-excitatory to output-excitatory
    pConn['ie_input'] = pConn['ee_input']  # Liquid-inhibitory to output-excitatory
    pConn['ei_input'] = 0.0   # Liquid-excitatory to output-inhibitory (NO CONNECTIONS)
    pConn['ee'] = 1.0         # Output-excitatory to output-excitatory (NO RECURRENT CONNECTIONS IN THE OUTPUT-EXCITATORY LAYER)
    pConn['ei'] = 0.0025      # Output-excitatory to output-inhibitory
    pConn['ie'] = 0.9         # Output-inhibitory to output-excitatory
    pConn['ii'] = 0.1         # Output-inhibitory to output-inhibitory (NO RECURRENT CONNECTIONS IN THE OUTPUT-INHIBITORY LAYER)

    weight = {}
    weight['ee_input'] = 0.3
    weight['ie_input'] = weight['ee_input']
    weight['ei_input'] = 0.2 
    weight['ee'] = 0.1
    weight['ei'] = 10.4
    weight['ie'] = 4.0
    weight['ii'] = 0.4

    tag_mode = True  # Assert if tag mode is enabled, wherein the LeAe and LiAe weight matrices
                     # need to be additionally written into the weights or weights_stoc directory.

    # Create random connections between liquid-excitatory(Le) and output-excitatory(Ae) neurons
    print 'Create a random connection matrix between liquid-excitatory(Le) and output-excitatory(Ae) neurons'
    connNameList = ['LeAe']

    if(stoc_enable == 0):
       for name in connNameList:
           weightMatrix  = np.random.random((nLiquidE, nE)) + 0.01
           weightMatrix *= weight['ee_input']
           if pConn['ee_input'] < 1.0:
              weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
           else:
              weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nE) for i in xrange(nLiquidE)]
           print 'Save connection matrix', dataPath+name
           np.save(dataPath+name, weightList)
           if(tag_mode):
              print 'Save connection matrix', outDataPath+name, '\n'
              np.save(outDataPath+name, weightList)
           else:
              print
    else:
       for name in connNameList:
           weightMatrix  = np.random.random((nLiquidE, nE))
           weightMatrix  = (weightMatrix < pConn['ee_input']) * 1.0
           weightMatrix += ((1.0-weightMatrix) * weight['ee_input'])
           # if(debug_mode):
              # np.set_printoptions(threshold='nan')
              # print weightMatrix
           weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nE) for i in xrange(nLiquidE)]
           print 'Save connection matrix', dataPath+name
           np.save(dataPath+name, weightList)
           if(tag_mode):
              print 'Save connection matrix', outDataPath+name, '\n'
              np.save(outDataPath+name, weightList)
           else:
              print

    # Create random connections between liquid-inhibitory(Li) and output-excitatory(Ae) neurons
    print 'Create a random connection matrix between liquid-inhibitory(Li) and output-excitatory(Ae) neurons'
    connNameList = ['LiAe']

    if(stoc_enable == 0):
       for name in connNameList:
           weightMatrix = np.random.random((nLiquidI, nE)) + 0.01
           weightMatrix *= weight['ie_input']
           if pConn['ie_input'] < 1.0:
              weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie_input'])
           else:
              weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nE) for i in xrange(nLiquidI)]
           print 'Save connection matrix', dataPath+name
           np.save(dataPath+name, weightList)
           if(tag_mode):
              print 'Save connection matrix', outDataPath+name, '\n'
              np.save(outDataPath+name, weightList)
           else:
              print
    else:
       for name in connNameList:
           weightMatrix  = np.random.random((nLiquidI, nE))
           weightMatrix  = (weightMatrix < pConn['ie_input']) * 1.0
           weightMatrix += ((1.0-weightMatrix) * weight['ie_input'])
           # if(debug_mode):
              # np.set_printoptions(threshold='nan')
              # print weightMatrix
           weightList = [(i, j, weightMatrix[i,j]) for j in xrange(nE) for i in xrange(nLiquidI)]
           print 'Save connection matrix', dataPath+name
           np.save(dataPath+name, weightList)
           if(tag_mode):
              print 'Save connection matrix', outDataPath+name, '\n'
              np.save(outDataPath+name, weightList)
           else:
              print

    # Create synaptic connections between output-excitatory(Ae) and output-inhibitory(Ai) neurons'
    print 'Create a connection matrix between output-excitatory(Ae) and output-inhibitory(Ai) neurons'
    connNameList = ['AeAi']
    for name in connNameList:
        if nE == nI:
            weightList = [(i, i, weight['ei']) for i in xrange(nE)]
        else:
            weightMatrix = np.random.random((nE, nI))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print 'Save connection matrix', dataPath+name
        np.save(dataPath+name, weightList)
        if(tag_mode):
           print 'Save connection matrix', outDataPath+name, '\n'
           np.save(outDataPath+name, weightList)
        else:
           print

    # Create synaptic connections between output-inhibitory(Ai) and output-excitatory(Ae) neurons'
    print 'Create a connection matrix between output-inhibitory(Ai) and output-excitatory(Ae) neurons'
    connNameList = ['AiAe']
    for name in connNameList:
        if nE == nI:
            weightMatrix = np.ones((nI, nE))
            weightMatrix *= weight['ie']
            for i in xrange(nI):
                weightMatrix[i,i] = 0
            weightList = [(i, j, weightMatrix[i,j]) for i in xrange(nI) for j in xrange(nE)]
        else:
            weightMatrix = np.random.random((nI, nE))
            weightMatrix *= weight['ie']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print 'Save connection matrix', dataPath+name
        np.save(dataPath+name, weightList)
        if(tag_mode):
           print 'Save connection matrix', outDataPath+name, '\n'
           np.save(outDataPath+name, weightList)
        else:
           print

if __name__ == "__main__":
    # Parse the command line arguments
    stoc_enable = 0
    debug_mode  = 0
    opts, args = getopt.getopt(sys.argv[1:],"hsd",["help", "stoc_enable", "debug"])

    for opt,arg in opts:
      if opt in ("-h", "--help"):
         print '---------------'
         print 'Usage Example:'
         print '---------------'
         print os.path.basename(__file__) + ' --help        -> Print script usage example'
         print os.path.basename(__file__) + ' --stoc_enable -> Enable Stochasticity'
         print os.path.basename(__file__) + ' --debug       -> Enable debug mode'
         sys.exit(1)

      elif opt in ("-s", "--stoc_enable"): 
         stoc_enable = 1

      elif opt in ("-d", "--debug"): 
         debug_mode  = 1

    if(debug_mode):
       print '#################### DEBUG MODE ACTIVATED! ####################'

    if(stoc_enable):
       print '----------------------------------------------------------------------------'
       print 'Synapses connecting the liquid and output-excitatory neurons are STOCHASTIC!'
       print '----------------------------------------------------------------------------'
    else:
       print '--------------------------------------------------------------------------------'
       print 'Synapses connecting the liquid and output-excitatory neurons are NOT STOCHASTIC!'
       print '--------------------------------------------------------------------------------'

    # Initialize the synaptic weight matrices
    create_weights()

