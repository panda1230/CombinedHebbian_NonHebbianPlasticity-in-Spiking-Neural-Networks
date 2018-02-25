'''
Paper Title: Learning to Generate Sequences with Combination of Hebbian and Non-hebbian Plasticity in Recurrent Spiking Neural Networks
https://www.frontiersin.org/articles/10.3389/fnins.2017.00693/full
@author: Priyadarshini Panda and Kaushik Roy
This code uses the combined plasticity learning to character recognition using a spiking reservoir without output/readout neurons. 
'''


import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy
import cPickle as pickle
from struct import unpack
import os
import sys
import getopt

import scipy.io as sio

#--------------------------------------------------------------------------
# set parameters and equations
#--------------------------------------------------------------------------
import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging
import brian as b
from brian import *
import math


b.set_global_preferences(
                        defaultclock          = b.Clock(dt=0.5*b.ms), # The default clock to use if none is provided or defined in any enclosing scope.
                        useweave              = True,  # Defines whether or not functions should use inlined compiled C code where defined.
                        gcc_options           = ['-ffast-math -march=native'], # Defines the compiler switches passed to the gcc compiler.
                        #For gcc versions 4.2+ we recommend using -march=native. By default, the -ffast-math optimizations are turned on
                        usecodegen            = True,  # Whether or not to use experimental code generation support.
                        usecodegenweave       = True,  # Whether or not to use C with experimental code generation support.
                        usecodegenstateupdate = True,  # Whether or not to use experimental code generation support on state updaters.
                        usecodegenthreshold   = False, # Whether or not to use experimental code generation support on thresholds.
                        usenewpropagate       = True,  # Whether or not to use experimental new C propagation functions.
                        usecstdp              = False, # Whether or not to use experimental new C STDP.
                       )
#--------------------------------------------------------------------------
# Specify the location of data
#-------------------------------------------------------------------------
DARPA_data_path = '/data/'
#--------------------------------------------------------------------------
# User-defined functions
#--------------------------------------------------------------------------

def get_DARPA_data():
    num_img=1000
    #test = sio.loadmat(caltech_data_path+'butterfly_grayscale.mat')
    nI   = 784  # Input image dimension [28x28x1]
    j=0
    bg=np.array(['A1','A2','R','C','O','T','F'])
    data   = np.zeros((num_img, nI), dtype=np.uint8)
    labels = np.zeros((num_img),     dtype=np.uint8)
    

    for i in range(num_img):
        img_name = str(bg[j]) + '.mat'
        test      = sio.loadmat(DARPA_data_path + img_name)
        if j<3:
           img = test[str(bg[j])]
        else:
           img = test['image1']
        img=np.uint8(img)
        if j<3:
           img=img/3
        else:
           img=img*3
          
        
        # print img_name
        # print np.shape(img)  # (32, 32, 3)
        # imgnew = img[:,:,0]
        # imgnew = imgnew.reshape((nI))
        # diff = img[:,:,0].reshape((nI)) - imgnew
        # print '######', np.sum(diff), '######'
        #img = img[:,:,CHNL] * (img[:,:,CHNL]>=PXL_THRESH)
        # np.set_printoptions(threshold='nan')
        # print img
        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()
        img = img.reshape((nI))
        data[i,:] = np.copy(img)
        labels[i] = j
        j += 1
        if j ==7: #num_images =5
            j=0

    imgdata = {'data':data, 'labels':labels}
    return imgdata


def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4

    # Determine the number of rows of the target matrix
    if fileName[-4-offset] == 'X':
       n_src = n_input
    elif fileName[-4-offset] == 'L':
       if fileName[-3-offset]=='e':
          n_src = n_liquid_e
       else:
          n_src = n_liquid_i
    else:
       if fileName[-3-offset]=='e':
          n_src = n_e
       else:
          n_src = n_i

    # Determine the number of columns of the target matrix
    if fileName[-2-offset] == 'L':
       if fileName[-1-offset]=='e':
          n_tgt = n_liquid_e
       else:
          n_tgt = n_liquid_i
    else:
       if fileName[-1-offset]=='e':
          n_tgt = n_e
       else:
          n_tgt = n_i
    readout = np.load(fileName)
    # print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
       value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr

def save_connections(ending = ''):
    for connName in save_conns:  # XeLe, XeLi, LeLe, LeLi, LiLe, LiLi, LeAe, LiAe, AeAi, AiAe
        connMatrix = connections[connName][:]
        connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
        np.save(store_weight_path + connName + ending, connListSparse)

def save_theta(ending = ''):
    # Save the adapted threshold of the output-excitatory neurons
    for pop_name in population_names:  # population_names: A
        np.save(store_weight_path + 'theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

    # Save the adapted threshold of the liquid-excitatory and inhibitory neurons
    for pop_name in liquid_population_names:   # liquid_population_names: L
        for connType in recurrent_conn_names:  # recurrent_conn_names   : ei, ie
            np.save(store_weight_path + 'theta_' + pop_name + connType[0] + ending, neuron_groups[pop_name + connType[0]].theta)

def save_assignments(ending = ''):
    np.save(store_weight_path + 'assignments' + ending, assignments)

def save_postlabel(ending = ''):
    np.save(store_weight_path + 'assignments' + ending, post_label)

def get_2d_weights_liquid_output():
    conn_name      = ['LeAe', 'LiAe']
    weight_matrix  = np.zeros((n_liquid, n_e))
    n_e_sqrt       = int(np.sqrt(n_e))
    n_in_sqrt      = int(np.sqrt(n_liquid))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    offset = 0
    for name in conn_name:
        connMatrix = connections[name][:]
        if(name[1] == 'e'):
           n_liquid_use = n_liquid_e
        else:
           n_liquid_use = n_liquid_i

        # Load the weight matrix
        for i in xrange(n_liquid_use):
            weight_matrix[offset+i, connMatrix.rowj[i]] = connMatrix.rowdata[i]

        # Update the offset used to index the weight matrix
        offset = n_liquid_use

    # Form the rearranged weight matrix
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
            rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights

def plot_2d_weights_liquid_output():
    name = 'Liquid(Le,Li) to Output-Excitatory(Ae)'
    weights = get_2d_weights_liquid_output()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('Weights of connection: ' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_weights_liquid_output(im, fig):
    weights = get_2d_weights_liquid_output()
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_2d_weights_liquid_liquid():
    name      = 'LeLe'
    weight_matrix  = np.zeros((n_liquid_e, n_liquid_e))
    n_e_sqrt       = int(np.sqrt(n_liquid_e))
    n_in_sqrt      = int(np.sqrt(n_liquid_e))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    offset = 0
    connMatrix = connections[name][:]
    print num_values_col
    for i in xrange(n_liquid_e):
        weight_matrix[i, connMatrix.rowj[i]] = connMatrix.rowdata[i]

    print np.shape(weight_matrix)
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights
#    for name in conn_name:
#        connMatrix = connections[name][:]
#        if(name[1] == 'e'):
#           n_liquid_use = n_liquid_e
#        else:
#           n_liquid_use = n_liquid_i

        # Load the weight matrix
#        for i in xrange(n_liquid_use):
#            weight_matrix[offset+i, connMatrix.rowj[i]] = connMatrix.rowdata[i]

        # Update the offset used to index the weight matrix
#        offset = n_liquid_use
#    print np.shape(weight_matrix)
    # Form the rearranged weight matrix
#    for i in xrange(n_e_sqrt):
#        for j in xrange(n_e_sqrt):
#            rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
#    return rearranged_weights



def plot_2d_weights_liquid_liquid():
    name = 'Liquid(Le, Li) to Liquid(Le)'
    weights = get_2d_weights_liquid_liquid()
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('Weights of connection: ' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_weights_liquid_liquid(im, fig):
    weights = get_2d_weights_liquid_liquid()
    im.set_array(weights)
    fig.canvas.draw()
    return im




def get_2d_weights_input_liquid(print_xeli=True):
    if(print_xeli):
       conn_name   = ['XeLe', 'XeLi']
    else:
       conn_name   = ['XeLe']
    weight_matrix  = np.zeros((n_input, n_liquid))
    n_e_sqrt       = int(np.sqrt(n_liquid))
    n_in_sqrt      = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    offset = 0
    for name in conn_name:
        connMatrix = connections[name][:]
        if(name[-1] == 'e'):
           n_liquid_use = n_liquid_e
        else:
           n_liquid_use = n_liquid_i

        # Load the weight matrix
        # weight_matrix[:, offset+np.arange(n_liquid_use)] = connMatrix.toarray()
        for i in xrange(n_input):
            weight_matrix[i, offset+connMatrix.rowj[i]] = connMatrix.rowdata[i]

        # Update the offset used to index the weight matrix
        offset = n_liquid_use

    # Form the rearranged weight matrix
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
            rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights

def plot_2d_weights_input_liquid(print_xeli=True):
    name = 'Input(Xe) to Liquid(Le,Li)'
    weights = get_2d_weights_input_liquid(print_xeli)
    fig = b.figure(fig_num, figsize = (18, 18))
    im2 = b.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('Weights of connection: ' + name)
    fig.canvas.draw()
    return im2, fig

def update_2d_weights_input_liquid(im, fig, print_xeli=True):
    weights = get_2d_weights_input_liquid(print_xeli)
    im.set_array(weights)
    fig.canvas.draw()
    return im

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval) -1
    start_num = current_example_num - update_interval
    end_num = current_example_num
    #for idx in range(start_num, end_num):
        #print '===================================='
        #print str(idx) + '. ' + str(input_numbers[idx])
        #print '------'
        #print outputNumbers[idx, :]
        #print '===================================='
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def plot_performance(fig_num):
    num_evaluations = int(math.ceil(num_examples/float(update_interval)))
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = b.figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    b.ylim(ymax = 100)
    b.title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    max_rates = [0] * 10
    for i in xrange(7):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
            max_rates[i] = np.amax(spike_rates[assignments == i])
    if(test_mode):
       print 'summed_rates:', summed_rates
       # print 'Sorted summed rates:', np.argsort(summed_rates)[::-1]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    if(tag_mode):
       n_e_use = n_e
    else:
       n_e_use = n_liquid_e
    assignments         = np.zeros(n_e_use)
    assignments_alldig  = np.ones((n_e_use, 10)) * -1 #
    input_nums          = np.asarray(input_numbers)
    maximum_rate        = [0] * n_e_use
    maximum_rate_alldig = np.zeros((n_e_use, 10))     #
    count_assign        = [0] * n_e_use               #
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
            maximum_rate_alldig[:,j] = rate
            for i in xrange(n_e_use):
                if rate[i] > maximum_rate[i]:
                   # print str(i) + ': [Old] assign: ' + str(assignments[i]) #
                   # print maximum_rate[i]                           #
                   # diff = rate[i] - maximum_rate[i]                #
                   ## maximum_rate_alldig[i, count_assign[i]] = rate[i] #
                   ## assignments_alldig [i, count_assign[i]] = j       #
                   ## count_assign[i] += 1                              #
                   maximum_rate[i]  = rate[i]
                   assignments[i]   = j
                   # print str(i) + ': [new] assign: ' + str(assignments[i]) #
                   # print maximum_rate[i] #
                   # print diff            #
                   # print '\n'            #
    assignments_alldig       = np.argsort(-maximum_rate_alldig)      #
    maximum_rate_alldig_sort = np.sort   (-maximum_rate_alldig) * -1 #
    assignments_alldig[np.where(maximum_rate_alldig_sort==0)] = -1   #

    if(not test_mode and not tag_mode):
       np.save(store_weight_path + 'assignments_alldig_Le'  + ending, assignments_alldig)
       np.save(store_weight_path + 'maximum_rate_alldig_Le' + ending, maximum_rate_alldig)
    return assignments


#--------------------------------------------------------------------------
# Parse command line arguments
#--------------------------------------------------------------------------
stoc_enable = 0
opts, args  = getopt.getopt(sys.argv[1:],"hs",["help", "stoc_enable"])

for opt,arg in opts:
  if opt in ("-h", "--help"):
     print '---------------'
     print 'Usage Example:'
     print '---------------'
     print os.path.basename(__file__) + ' --help        -> Print script usage example'
     print os.path.basename(__file__) + ' --stoc_enable -> Enable Stochasticity'
     sys.exit(1)

  elif opt in ("-s", "--stoc_enable"): 
     stoc_enable = 1

if(stoc_enable):
   print '------------------------------------------------------------------------'
   print 'Synapses connecting liquid and output-excitatory neurons are STOCHASTIC!'
   print '------------------------------------------------------------------------'
else:
   print '----------------------------------------------------------------------------'
   print 'Synapses connecting liquid and output-excitatory neurons are NOT STOCHASTIC!'
   print '----------------------------------------------------------------------------'

#--------------------------------------------------------------------------
# Load dataset
#--------------------------------------------------------------------------
start = time.time()
training = get_DARPA_data()
end = time.time()
# print 'time needed to load training set:', end - start

start = time.time()
#testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
testing = get_DARPA_data()
end = time.time()
# print 'time needed to load test set:', end - start


#--------------------------------------------------------------------------
# LSM-SNN topology parameters
#--------------------------------------------------------------------------
data_path  = './'
ending     = ''
n_input    = 784
n_label    = 1000
n_liquid   = 200
n_liquid_e = int(ceil(0.80*n_liquid))
n_liquid_i = n_liquid - n_liquid_e
n_e        = 60
n_i        = n_e
n_output   = 10

#--------------------------------------------------------------------------
# Initialize a seed for the random number generators
#--------------------------------------------------------------------------
np.random.seed(0)

#--------------------------------------------------------------------------
# SNN simulation parameters
#--------------------------------------------------------------------------
num_examples        = 100 * 1
single_example_time = 0.35 * b.second
resting_time        = 0.15 * b.second
runtime             = num_examples * (single_example_time + resting_time)
dt_clock            = 0.5 * b.ms  # Need to change the default clock option in global parameters
num_timesteps       = single_example_time / dt_clock
use_testing_set       = False
test_mode             = True
tag_mode              = False

if num_examples <= 10000:
    weight_update_interval    = 1
    save_connections_interval = 10000
else:
    weight_update_interval    = 100
    save_connections_interval = 10000

if test_mode:
    if(stoc_enable == 0):
       load_weight_path  = data_path + 'weights/'


    do_plot_performance = True
    record_spikes       = True
    record_state        = False
    ee_STDP_on          = False
    lqd_op_STDP_on      = False
    prune_mode          = False
    update_interval     = 100

elif tag_mode:
    if(stoc_enable == 0):
       load_weight_path  = data_path + 'weights/'
       store_weight_path = data_path + 'weights/'


    do_plot_performance = False
    record_spikes       = True
    record_state        = False
    ee_STDP_on          = False
    lqd_op_STDP_on      = True
    prune_mode          = False
    update_interval     = num_examples

else:
    if(stoc_enable == 0):
       load_weight_path  = data_path + 'random/'
       store_weight_path = data_path + 'weights/'


    do_plot_performance = True
    record_spikes       = True
    record_state        = False
    ee_STDP_on          = True
    lqd_op_STDP_on      = False
    update_interval     = num_examples

# Input spike intensity configuration
input_intensity       = 2.
start_input_intensity = input_intensity

#--------------------------------------------------------------------------
# Create target labels for the output-excitatory neurons to implement the
# forced learning algorithm
#--------------------------------------------------------------------------
if not test_mode:
   train_dig    = np.array([0, 3, 8]) #np.array([0, 1, 6, 9, 2, 4, 7, 3, 8])
   n_output_use = train_dig.size
   assert(n_e%n_output_use == 0)
   neurons_per_op = n_e/n_output_use
 # post_indx      = np.random.permutation(n_e)
   post_indx      = np.arange(n_e)
   post_label     = np.zeros(n_e)

   for i in xrange(n_output_use):
       post_label[post_indx[i*neurons_per_op:(i+1)*neurons_per_op]] = train_dig[i]

   print '##### Output-excitatory neuron post-label #####'
   print post_label

#--------------------------------------------------------------------------
# LIF liquid-excitatory neuron
#--------------------------------------------------------------------------
v_rest_le   = -65. * b.mV
v_reset_le  = -65. * b.mV
v_thresh_le = -52. * b.mV
refrac_le   =  5.  * b.ms

if test_mode or tag_mode:
   scr_le        = 'v = v_reset_le; timer = 0*ms'
else:
   tc_theta_le   = 1e8 * b.ms
   theta_plus_le = 0.1 * b.mV
   scr_le        = 'v = v_reset_le; theta += theta_plus_le; timer = 0*ms'

v_thresh_le = '(v>(theta + ' + str(v_thresh_le) + ')) * (timer>refrac_le)'

neuron_eqs_le = '''
        dv/dt  = ((v_rest_le - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *          -v                                : amp
        I_synI = gi * nS * (-100.*mV-v)                               : amp
        dge/dt = -ge/(1.0*ms)                                         : 1
        dgi/dt = -gi/(2.0*ms)                                         : 1
        '''

if test_mode or tag_mode:
   neuron_eqs_le += '\n  theta  : volt'
else:
   neuron_eqs_le += '\n  dtheta/dt = -theta / (tc_theta_le)  : volt'

neuron_eqs_le += '\n  dtimer/dt = 100.0  : ms'

#--------------------------------------------------------------------------
# LIF liquid-inhibitory neuron
#--------------------------------------------------------------------------
v_rest_li   = -60. * b.mV
v_reset_li  = -45. * b.mV
v_thresh_li = -40. * b.mV
refrac_li   =  2.  * b.ms


neuron_eqs_li = '''
        dv/dt  = ((v_rest_li - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                                : amp
        I_synI = gi * nS * (-85.*mV-v)                               : amp
        dge/dt = -ge/(1.0*ms)                                        : 1
        dgi/dt = -gi/(2.0*ms)                                        : 1
        '''

if test_mode or tag_mode:
   neuron_eqs_li += '\n  theta  : volt'
else:
   neuron_eqs_li += '\n  theta  : volt'
 

#--------------------------------------------------------------------------
# Implement STDP for synapses connecting the input and liquid-excitatory neurons
#--------------------------------------------------------------------------
# Synaptic potentiation parameters
tc_pre_xe      = 20*b.ms
pre_rst_xe     = 0.01 #0.005
nu_post_le     = pre_rst_xe
STDP_offset_le = 0.4
exp_post_le    = 0.9

# Synaptic depression parameters
tc_post_le  = 20*b.ms
post_rst_le = 0.001

# Synaptic weight constraints
wmax_xele = 1.0
wmin_xele = 0.0

# STDP equations
eqs_stdp_xele = '''
        dpre/dt  = -pre/(tc_pre_xe)    : 1.0
        dpost/dt = -post/(tc_post_le)  : 1.0
        
        '''
if(stoc_enable == 0):
   eqs_stdp_pre_xe  = 'pre += 1.'
   eqs_stdp_post_le = 'w += (nu_post_le * (pre - STDP_offset_le) * ((wmax_xele - w)**exp_post_le)); post += 1.'
   eqs_stdp_lele = eqs_stdp_post_le
   eqs_stdp_lile = 'w += (-nu_post_le * (pre - STDP_offset_le) * ((wmax_xele - w)**exp_post_le)); post += 1.'
else:
   eqs_stdp_pre_xe  = 'w -= ((post>rand(' + str(n_liquid_e) + '))*1.0); pre = pre_rst_xe'
   eqs_stdp_post_le = 'w += ((pre>rand(' + str(n_input) + '))*1.0); post = post_rst_le'

#--------------------------------------------------------------------------
# LIF output-excitatory neuron
#--------------------------------------------------------------------------
v_rest_e     = -65. * b.mV
v_reset_e    = -65. * b.mV
v_thresh_e   = -60. * b.mV
refrac_e     =  refrac_le
lqd_sample_t =  10. # Unit: ms

if test_mode:
   scr_e        = 'v = v_reset_e; timer = 0*ms'
else:
   tc_theta_e   = 1e8 * b.ms
   theta_plus_e = 0.1 * b.mV
   scr_e        = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

v_thresh_e = '(v>(theta + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

neuron_eqs_e = '''
        dv/dt  = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *  -v                                       : amp
        I_synI = gi * nS * (-100.*mV-v)                              : amp
        dge/dt = -ge/(4.0*ms)                                        : 1
        dgi/dt = -gi/(2.0*ms)                                        : 1
        '''

if test_mode:
   neuron_eqs_e += '\n  theta  : volt'
else:
   neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta_e)  : volt'
   neuron_eqs_e += '\n  img_label                          : 1.0'
   neuron_eqs_e += '\n  post_label                         : 1.0'

neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'
neuron_eqs_e += '\n  dtcount/dt = 1000.0  : 1.0'

#--------------------------------------------------------------------------
# LIF output-inhibitory neuron
#--------------------------------------------------------------------------
v_rest_i   = -60. * b.mV
v_reset_i  = -45. * b.mV
v_thresh_i = -40. * b.mV
refrac_i   =  refrac_li

neuron_eqs_i = '''
        dv/dt  = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)  : volt
        I_synE = ge * nS *         -v                               : amp
        I_synI = gi * nS * (-85.*mV-v)                              : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        '''

#--------------------------------------------------------------------------
# Implement STDP for synapses connecting the liquid and output-excitatory neurons
#--------------------------------------------------------------------------
# Stochastic-STDP potentiation
tc_pre_ee   = 20*b.ms
pre_rst     = 0.005
nu_ee_post  = pre_rst
exp_ee_post = 0.9
STDP_offset = 0.4

# Stochastic-STDP depression
tc_post_1_ee = 20*b.ms
post_rst     = 0.005

# Synaptic weight constraints
wmax_ee = 1.0
wmin_ee = 0.0

eqs_stdp_ee = '''
            dpre/dt    = -pre/(tc_pre_ee)      : 1.0
            dpost/dt   = -post/(tc_post_1_ee)  : 1.0
            img_label                          : 1.0
            dtcount/dt =  1000                 : 1.0
            '''
if(stoc_enable == 0):
   eqs_stdp_pre_ee  = 'pre += 1.'
   eqs_stdp_post_ee = 'tcount += 0; img_label += 0; w += (nu_ee_post * (pre - STDP_offset) * ((wmax_ee - w)**exp_ee_post)); post += 1.'

   # The exponential-STDP post-equations are identical for both the liquid-excitatory and inhibitory neurons
   eqs_stdp_post_ae_le = eqs_stdp_post_ee
   eqs_stdp_post_ae_li = eqs_stdp_post_ee

#--------------------------------------------------------------------------
# LSM-SNN connectivity specification
#--------------------------------------------------------------------------
conn_structure              = 'sparse'
delay                       = {}
input_population_names      = ['X']
liquid_population_names     = ['L']
population_names            = ['A']
input_connection_names      = ['XL']
output_connection_names     = ['LA']
save_conns                  = ['XeLe', 'XeLi',                 \
                               'LeLe', 'LeLi', 'LiLe', 'LiLi', \
                               'LeAe', 'LiAe', 'AeAi', 'AiAe']
input_conn_names            = ['ee_input', 'ei_input']
output_conn_names           = ['ee_output', 'ie_output']
recurrent_conn_names        = ['ei', 'ie']
liquid_recurrent_conn_names = ['ee', 'ei', 'ie', 'ii']
delay['ee_input']           = (0*b.ms,10*b.ms)
delay['ei_input']           = (0*b.ms,5*b.ms)
delay['ee'] = (0*b.ms,5*b.ms)
delay['ei'] = (0*b.ms,1*b.ms)
delay['ie'] = (0*b.ms,1*b.ms)
delay['ii'] = (0*b.ms,1*b.ms)

#--------------------------------------------------------------------------
# Create the neuron groups
#--------------------------------------------------------------------------
b.ion()
fig_num        = 1
neuron_groups  = {}
input_groups   = {}
connections    = {}
stdp_methods   = {}
rate_monitors  = {}
spike_monitors = {}
spike_counters = {}
if(test_mode or tag_mode):
   result_monitor = np.zeros((update_interval,n_liquid_e))
else:
   result_monitor = np.zeros((update_interval,n_liquid_e))
Ipost_monitors = {}
theta_monitors = {}
ge_monitors    = {}
gi_monitors    = {}
v_monitors     = {}

# Liquid-excitatory and inhibitory neuron groups
neuron_groups['Liquide'] = b.NeuronGroup(n_liquid_e, neuron_eqs_le, threshold= v_thresh_le, refractory= refrac_le, reset= scr_le,
                                         compile = True, freeze = True)
neuron_groups['Liquidi'] = b.NeuronGroup(n_liquid_i, neuron_eqs_li, threshold= v_thresh_li, refractory= refrac_li, reset= v_reset_li,
                                         compile = True, freeze = True)

# Output-excitatory and inhibitory neuron groups
neuron_groups['e'] = b.NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e,
                                   compile = True, freeze = True)
neuron_groups['i'] = b.NeuronGroup(n_i*len(population_names), neuron_eqs_i, threshold= v_thresh_i, refractory= refrac_i, reset= v_reset_i,
                                   compile = True, freeze = True)

#--------------------------------------------------------------------------
# Create liquid population and recurrent connections
#--------------------------------------------------------------------------
for name in liquid_population_names:  # liquid_population_names: L
    neuron_groups[name+'e'] = neuron_groups['Liquide'].subgroup(n_liquid_e)
    neuron_groups[name+'i'] = neuron_groups['Liquidi'].subgroup(n_liquid_i)

    neuron_groups[name+'e'].v = v_rest_le - 40. * b.mV
    neuron_groups[name+'i'].v = v_rest_li - 40. * b.mV

    if test_mode or tag_mode or load_weight_path[-8:] == 'weights/':
        neuron_groups['Liquide'].theta = np.load(load_weight_path + 'theta_' + name + 'e' + ending + '.npy')
        neuron_groups['Liquidi'].theta = np.load(load_weight_path + 'theta_' + name + 'i' + ending + '.npy')
        print '\n------- LIQUID_EXCITATORY THETA --------'
        print neuron_groups['Liquide'].theta
        print '\n------- LIQUID_INHIBITORY THETA --------'
        print neuron_groups['Liquidi'].theta, '\n'

        theta_avg_liquide = np.mean(neuron_groups['Liquide'].theta)
        theta_avg_liquidi = np.mean(neuron_groups['Liquidi'].theta)
        print 'Average theta of the liquid-excitatory neurons =', theta_avg_liquide
        print 'Average theta of the liquid-inhibitory neurons =', theta_avg_liquidi, '\n'
    else:
        # Initialize theta (neuronal firing threshold adaptation parameter)
        neuron_groups['Liquide'].theta = np.ones ((n_liquid_e)) * 5.0*b.mV
        neuron_groups['Liquidi'].theta = np.zeros((n_liquid_i))

    # Create recurrent connections among neurons in the liquid layer'
    for conn_type in liquid_recurrent_conn_names:  # liquid_recurrent_conn_names: ee, ei, ie, ii
        connName = name+conn_type[0]+name+conn_type[1]
        print '########## Creating connection:' + connName + ' ##########'
        weightMatrix = get_matrix_from_file(load_weight_path + connName + ending + '.npy')
        weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
        connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure,
                                             state = 'g'+conn_type[0])
        connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)
 
        # Enable synaptic plasticity for the input to liquid-excitatory connections
        if ee_STDP_on:
           print '########## Enabling synaptic plasticity:' + connName + ' ##########'
           if (connName == 'LiLe'): # or (connName == 'LeLe'):
              stdp_methods[connName] = b.STDP(connections[connName], eqs= eqs_stdp_xele, pre= eqs_stdp_pre_xe,
                                              post= eqs_stdp_lile, wmin= wmin_xele, wmax= wmax_xele)
           if (connName == 'LeLe'): # or (connName == 'LeLe'):
              stdp_methods[connName] = b.STDP(connections[connName], eqs= eqs_stdp_xele, pre= eqs_stdp_pre_xe,
                                              post= eqs_stdp_lele, wmin= wmin_xele, wmax= wmax_xele)


    # Create rate and spike monitors
    rate_monitors[name+'e']  = b.PopulationRateMonitor(neuron_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
    rate_monitors[name+'i']  = b.PopulationRateMonitor(neuron_groups[name+'i'], bin = (single_example_time+resting_time)/b.second)
    spike_counters[name+'e'] = b.SpikeCounter(neuron_groups[name+'e'])
    spike_counters[name+'i'] = b.SpikeCounter(neuron_groups[name+'i'])

    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b.SpikeMonitor(neuron_groups[name+'i'])

if record_spikes:
    b.figure(fig_num)
    fig_num += 1
    b.ion()
    b.subplot(211)
    b.raster_plot(spike_monitors['Le'], refresh=1000*b.ms, showlast=1000*b.ms)
    b.subplot(212)
    b.raster_plot(spike_monitors['Li'], refresh=1000*b.ms, showlast=1000*b.ms)

#--------------------------------------------------------------------------
# Create input population and connections from input populations
#--------------------------------------------------------------------------
for i,name in enumerate(input_population_names):  # input_population_names: X
    if name == 'Y':
       input_groups[name+'e'] = b.PoissonGroup(n_label, 0)
    else:
       input_groups[name+'e'] = b.PoissonGroup(n_input, 0)
    rate_monitors[name+'e'] = b.PopulationRateMonitor(input_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)

for name in input_connection_names:    # input_connection_names: XL
    for connType in input_conn_names:  # input_conn_names      : ee_input, ei_input
        connName = name[0] + connType[0] + name[1] + connType[1]
        print '########## Creating connection:' + connName + ' ##########'
        weightMatrix = get_matrix_from_file(load_weight_path + connName + ending + '.npy')
        weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
        connections[connName] = b.Connection(input_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure,
                                             state = 'g'+connType[0], delay=True, max_delay=delay[connType][1])
        connections[connName].connect(input_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix, delay=delay[connType])

        # Enable synaptic plasticity for the input to liquid-excitatory connections
        if ee_STDP_on:
           print '########## Enabling synaptic plasticity:' + connName + ' ##########'
           if(connType[1] == 'e'):
              stdp_methods[connName] = b.STDP(connections[connName], eqs= eqs_stdp_xele, pre= eqs_stdp_pre_xe,
                                              post= eqs_stdp_post_le, wmin= wmin_xele, wmax= wmax_xele)

#--------------------------------------------------------------------------
# Create output network population with lateral inhibition
#--------------------------------------------------------------------------
for name in population_names:  # population_names: A
    neuron_groups[name+'e'] = neuron_groups['e'].subgroup(n_e)
    neuron_groups[name+'i'] = neuron_groups['i'].subgroup(n_i)

    neuron_groups[name+'e'].v = v_rest_e - 40. * b.mV
    neuron_groups[name+'i'].v = v_rest_i - 40. * b.mV

    if test_mode:
       neuron_groups['e'].theta = np.load(load_weight_path + 'theta_' + name + ending + '.npy')
       print '\n------- OUTPUT_EXCITATORY THETA --------'
       print neuron_groups['e'].theta, '\n'

       target_assignments = np.load(load_weight_path + 'assignments' + ending + '.npy')
       print '\n------- TARGET ASSIGNMENTS --------'
       print target_assignments

       print '\n------- ASSIGNMENT STATISTICS --------'
       num_target_assign = np.zeros(n_liquid_e)
       theta_avg         = np.zeros(n_liquid_e)
       for i in xrange(n_liquid_e):
           num_target_assign[i] = len(np.where(target_assignments == i)[0])
           if(num_target_assign[i] > 0):
              theta_avg[i]      = np.sum(neuron_groups['Liquide'].theta[target_assignments == i]) / num_target_assign[i]
       print num_target_assign
       print theta_avg
    else:
       # Initialize theta (neuronal firing threshold adaptation parameter)
       neuron_groups['e'].theta = np.ones((n_e)) * 5.0*b.mV

    # Create the connections from liquid to output-excitatory layer
    for op_name in output_connection_names:  # output_connection_names: LA
        for connType in output_conn_names:   # output_conn_names      : ee_output, ie_output
            connName = op_name[0] + connType[0] + op_name[1] + connType[1]
            print '########## Creating connection:' + connName + ' ##########'
            weightMatrix = get_matrix_from_file(load_weight_path + connName + ending + '.npy')

            if(tag_mode and (connType[0] == 'e')):
               print 'Updating', load_weight_path + connName, 'weight matrix to implement the forced learning algorithm'
               assign_Le_fname = load_weight_path + 'assignments_alldig_Le.npy'
               assign_Le       = np.load(assign_Le_fname)
               assign_Le_sort  = np.copy(assign_Le)
               # assign_Le_sort= np.sort(-assign_Le)*-1
               # np.set_printoptions(threshold='nan')
               # print assign_Le_sort
               for i in xrange(n_e):
                   weightMatrix[:,i] *= (((assign_Le_sort[:,0]==post_label[i]) | (assign_Le_sort[:,1]==post_label[i])) * 1.0)

            weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
            connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure,
                                                 state = 'g'+connType[1])
            connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)

            # Enable synaptic plasticity in the liquid to output connections
            if lqd_op_STDP_on:
               print '########## Enabling synaptic plasticity:' + connName + ' ##########'
               if(connType[0] == 'e'):
                  stdp_methods[connName] = b.STDP(connections[connName], eqs= eqs_stdp_ee, pre= eqs_stdp_pre_ee,
                                                  post= eqs_stdp_post_ae_le, wmin= wmin_ee, wmax= wmax_ee)
               else:
                  stdp_methods[connName] = b.STDP(connections[connName], eqs= eqs_stdp_ee, pre= eqs_stdp_pre_ee,
                                                  post= eqs_stdp_post_ae_li, wmin= wmin_ee, wmax= wmax_ee)

    # Create the lateral inhibitory connections
    for conn_type in recurrent_conn_names:  # recurrent_conn_names: ei, ie
        connName = name+conn_type[0]+name+conn_type[1]
        print '########## Creating connection:' + connName + ' ##########'
        weightMatrix = get_matrix_from_file(load_weight_path + connName + ending + '.npy')
        weightMatrix = scipy.sparse.lil_matrix(weightMatrix)
        connections[connName] = b.Connection(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], structure= conn_structure,
                                                    state= 'g'+conn_type[0])
        connections[connName].connect(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], weightMatrix)

    # Create rate and spike monitors
    rate_monitors[name+'e']  = b.PopulationRateMonitor(neuron_groups[name+'e'], bin = (single_example_time+resting_time)/b.second)
    rate_monitors[name+'i']  = b.PopulationRateMonitor(neuron_groups[name+'i'], bin = (single_example_time+resting_time)/b.second)
    spike_counters[name+'e'] = b.SpikeCounter(neuron_groups[name+'e'])

    if record_spikes:
        spike_monitors[name+'e'] = b.SpikeMonitor(neuron_groups[name+'e'])
        spike_monitors[name+'i'] = b.SpikeMonitor(neuron_groups[name+'i'])

    if record_state:
        ge_monitors['e'] = b.StateMonitor(neuron_groups['e'], 'I_synE', record=True)
        # gi_monitors['e']    = b.StateMonitor(neuron_groups['e'], 'I_synI', record=True)
        # v_monitors['e']     = b.StateMonitor(neuron_groups['e'], 'v', record=True)
        # Ipost_monitors['e'] = b.StateMonitor(neuron_groups['e'], 'I_post', record=True)
        # theta_monitors['e'] = b.StateMonitor(neuron_groups['e'], 'theta', record=True)
        # Ipost_monitors['i'] = b.StateMonitor(neuron_groups['i'], 'I_post', record=True)

#if record_spikes:
#    b.figure(fig_num)
#    fig_num += 1
#    b.ion()
#    b.subplot(211)
#    b.raster_plot(spike_monitors['Ae'], refresh=1000*b.ms, showlast=1000*b.ms)
#    b.subplot(212)
#    b.raster_plot(spike_monitors['Ai'], refresh=1000*b.ms, showlast=1000*b.ms)

#if record_state:
#    b.figure(fig_num)
#    fig_num += 1
#    b.ion()
#    ge_monitors['e'].plot(refresh=1000*b.ms, showlast=1000*b.ms)
    # gi_monitors['e'].plot(refresh=1000*b.ms, showlast=1000*b.ms)
    # v_monitors['e'].plot(refresh=1000*b.ms, showlast=1000*b.ms)
    # Ipost_monitors['e'].plot(refresh=1000*b.ms, showlast=1000*b.ms)
    # Ipost_monitors['i'].plot(refresh=1000*b.ms, showlast=1000*b.ms)
    # theta_monitors['e'].plot(refresh=1000*b.ms, showlast=1000*b.ms)
    

#------------------------------------------------------------------------------
##Weight decay of Le-Le connections to understand context
#------------------------------------------------------------------------------
if not test_mode and not tag_mode:
    @network_operation()
    def update_stdpvar():
        w_LeLe = connections['LeLe'][:]
        for i in range(n_liquid_e):
              w_LeLe_mat  = w_LeLe.get_col(i)
              #- (STDP_offset/(2**pre)) w_XeAe_mat * 0.0005 * nu_ee_post**2
              ###Linear Decay-----------------------------------------------------------------
              w_LeLe_mat += (-( 0.0005 * nu_ee_post**20/((stdp_methods['LeLe'].post[i]+1)* 2**((math.fabs(neuron_groups['Liquide'].theta[i]))*6)* (w_LeLe_mat**2 +1))))

#--------------------------------------------------------------------------
# Run the simulation
#--------------------------------------------------------------------------
# @network_operation
# def print_neuron_timer():
#     print (((neuron_groups['e'].tcount%lqd_sample_t)==0)*1.0)
# b.network_operation(print_neuron_timer)

previous_spike_count    = np.zeros(n_e)
previous_spike_count_le = np.zeros(n_liquid_e)
previous_spike_count_li = np.zeros(n_liquid_i)

if not test_mode:
   neuron_groups['Ae'].post_label = post_label
   if(tag_mode):
      assignments = np.zeros(n_e)
   else:
      assignments = np.zeros(n_liquid_e)
else:
   assignments = target_assignments

input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

if(not test_mode):
   if(not tag_mode):
      # Plot the weights from input to liquid layer
      weight_monitor_xe_leli, fig_weights_xe_leli = plot_2d_weights_input_liquid(print_xeli=False)
      fig_num += 1
      #weight_monitor_lile_le, fig_weights_lile_le = plot_2d_weights_liquid_liquid()
      #fig_num += 1
   else:
    # Plot the weights from liquid to output layer
    weight_monitor_leli_ae, fig_weights_leli_ae = plot_2d_weights_liquid_output()
    fig_num += 1

if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)

for i,name in enumerate(input_population_names):  # input_population_names: X
    input_groups[name+'e'].rate = 0

b.run(0)
j = 0
k = 0
epoch = 0
if(test_mode or tag_mode):
   SPIKE_THRESH = 5  # Minimum spike-count of the output-excitatory (Ae) neurons
else:
   SPIKE_THRESH = 5  # Minimum spike-count of the liquid-excitatory (Le) neurons

# Configure to train the SNN on a subset of the input patterns
train_digits = np.array([0, 1, 2, 3, 4, 5, 6]) #np.array([0, 1, 6, 9, 2, 4, 7, 3, 8])
dig_indx     = 0
img_target   = train_digits[dig_indx]
img_count    = 0
n_occrnces   = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

while j < (int(num_examples)):
    if test_mode:
       if use_testing_set:
          # rates     = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
          # img_label = testing['y'][j%10000][0]
          while((testing['labels'][k%1000] != img_target)):#0) and (testing['labels'][k] != 3) and (testing['labels'][k][0] != 8)):
              k += 1
          rates     = testing['data'][k%1000] / 8. *  input_intensity
          img_label = testing['labels'][k%1000]
       else:
          while(training['labels'][k%1000] != img_target):
              k += 1
          rates     = training['data'][k%1000] / 8. *  input_intensity
          img_label = training['labels'][k%1000]
    else:
       # rates     = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
       # img_label = training['y'][j%60000][0] 
       while(training['labels'][k%1000] != img_target):
           k += 1
       rates     = training['data'][k%1000] / 8. *  input_intensity
       img_label = training['labels'][k%1000] 

    input_groups['Xe'].rate = rates
    print 'run number:', j+1, 'of', int(num_examples), '; image label:', img_label

    # Update the image label used in output neuron group and STDP implementation
    if((not test_mode) and tag_mode):
       neuron_groups['Ae'].img_label  = img_label
       stdp_methods['LeAe'].img_label = img_label
       stdp_methods['LiAe'].img_label = img_label

    b.run(single_example_time)
    epoch += 1  # Increment the number of training epochs

    if j % update_interval == 0 and j > 0:
       if not test_mode:
          assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
          print assignments
       else:
          assignments = target_assignments
          print assignments

    if j % weight_update_interval == 0 and (not test_mode):
       if(not tag_mode):
          update_2d_weights_input_liquid(weight_monitor_xe_leli, fig_weights_xe_leli, print_xeli=True)
          #update_2d_weights_liquid_liquid(weight_monitor_lile_le, fig_weights_lile_le)
       else:
          update_2d_weights_liquid_output(weight_monitor_leli_ae, fig_weights_leli_ae)

    if j % save_connections_interval == 0 and j > 0 and not test_mode:
       save_connections(str(j))
       save_theta(str(j))
       save_assignments(str(j))

    # Update the spike counter of the output-excitatory neurons
    current_spike_count  = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])

    # Update the spike counter of the liquid-excitatory neurons
    current_spike_count_le  = np.asarray(spike_counters['Le'].count[:]) - previous_spike_count_le
    previous_spike_count_le = np.copy(spike_counters['Le'].count[:])

    # Update the spike counter of the liquid-inhibitory neurons
    current_spike_count_li  = np.asarray(spike_counters['Li'].count[:]) - previous_spike_count_li
    previous_spike_count_li = np.copy(spike_counters['Li'].count[:])

    if(test_mode or tag_mode):
       spike_count_use = np.sum(current_spike_count_le)
    else:
       spike_count_use = np.sum(current_spike_count_le)

    if(spike_count_use < SPIKE_THRESH):
       input_intensity += 1
       for i,name in enumerate(input_population_names):  # input_population_names: X
           input_groups[name+'e'].rate = 0
       b.run(resting_time)

       # Reset the timers used in output neuron group and STDP implementation
       neuron_groups['Ae'].tcount = np.zeros(n_e)

       if(lqd_op_STDP_on):
          stdp_methods['LeAe'].tcount = np.zeros(n_e)
          stdp_methods['LiAe'].tcount = np.zeros(n_e)
    else:
       if(test_mode or tag_mode):
          result_monitor[j%update_interval,:] = current_spike_count_le
       else:
          result_monitor[j%update_interval,:] = current_spike_count_le
       if test_mode and use_testing_set:
        # input_numbers[j] = testing['y'][j%10000][0]
          input_numbers[j] = testing['labels'][k%10000]
       else:
        # input_numbers[j] = training['y'][j%60000][0]
          input_numbers[j] = training['labels'][k%1000]
       outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
       if(test_mode):
          print 'Input  Label:', img_label
          print 'Output Label:', outputNumbers[j,0]
          print '----------------------------------'
       if j % update_interval == 0 and j > 0:
          if do_plot_performance:
             unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
           # print 'Classification performance', performance[:(j/float(update_interval))+1]
             print 'Classification performance', performance[:(j/float(update_interval))]
       for i,name in enumerate(input_population_names):  # input_population_names: X
           input_groups[name+'e'].rate = 0
       b.run(resting_time)
       input_intensity = start_input_intensity

       # Reset the timers used in output neuron group and STDP implementation
       neuron_groups['Ae'].tcount = np.zeros(n_e)

       if(lqd_op_STDP_on):
          stdp_methods['LeAe'].tcount = np.zeros(n_e)
          stdp_methods['LiAe'].tcount = np.zeros(n_e)

       if(test_mode and use_testing_set):
          j += 1
          k += 1
       else:
          # Show certain digits (eg. 5) multiple times
          j += 1
          img_count += 1
          if(img_count == n_occrnces[img_label]):
             img_count = 0
             k += 10

             # Update the image target
             dig_indx   = (dig_indx+1) % train_digits.size
             img_target = train_digits[dig_indx]

# Update classification performance at the end of the simulation
if do_plot_performance and test_mode:
   unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
 # print 'Classification performance', performance[:(j/float(update_interval))+1]
   print 'Classification performance', performance[:(j/float(update_interval))]

# Update synaptic weight plot at the end of the simulation
if(not test_mode):
   if(not tag_mode):
      update_2d_weights_input_liquid(weight_monitor_xe_leli, fig_weights_xe_leli, print_xeli=True)
      #update_2d_weights_liquid_liquid(weight_monitor_lile_le, fig_weights_lile_le)
   else:
      update_2d_weights_liquid_output(weight_monitor_leli_ae, fig_weights_leli_ae)

# Print the recently accessed image index
print 'Recently accessed image index (k) =', k

#--------------------------------------------------------------------------
# Save results
#--------------------------------------------------------------------------
if(not test_mode):
   save_theta()

   if(not tag_mode):
      print '\n------- LIQUID_EXCITATORY THETA --------'
      print neuron_groups['Liquide'].theta

      print '\n------- LIQUID_INHIBITORY THETA --------'
      print neuron_groups['Liquidi'].theta
   else:
      print '\n------- OUTPUT_EXCITATORY THETA --------'
      print neuron_groups['e'].theta

   # Tag the excitatory neurons based on their spiking rate
   update_interval = j
   assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
   if(tag_mode):
      print '\n------- OUTPUT-EXCITATORY NEURON ASSIGNMENTS --------'
   else:
      print '\n------- LIQUID-EXCITATORY NEURON ASSIGNMENTS --------'
   print assignments
   save_assignments()

   print '\n------- NUM ASSIGNMENTS --------'
   num_assign = np.zeros(n_output)
   for i in xrange(n_output):
       num_assign[i] = len(np.where(assignments == i)[0])
   print num_assign

if(not test_mode):
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)

#--------------------------------------------------------------------------
# plot results
#--------------------------------------------------------------------------
#if rate_monitors:
#    b.figure(fig_num)
#    fig_num += 1
#    for i, name in enumerate(rate_monitors):
#        b.subplot(len(rate_monitors), 1, i)
#        b.plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
#        b.title('Rates of population ' + name)

#if spike_monitors:
#    b.figure(fig_num)
#    fig_num += 1
#    for i, name in enumerate(spike_monitors):
#        b.subplot(len(spike_monitors), 1, i)
#        b.raster_plot(spike_monitors[name])
#        b.title('Spikes of population ' + name)

#if spike_counters:
#    b.figure(fig_num)
#    fig_num += 1
#    for i, name in enumerate(spike_counters):
#        b.subplot(len(spike_counters), 1, i)
#        b.plot(spike_counters['Ae'].count[:])
#        b.title('Spike count of population ' + name)

# print 'Classification performance', performance[len(performance)-1]
# plot_2d_input_weights()
b.ioff()
b.show()
