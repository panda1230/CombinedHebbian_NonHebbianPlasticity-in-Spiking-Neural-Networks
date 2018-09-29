# Combined Hebbian/Non-Hebbian Plasticity for Recognition in Recurrent Spiking Neural Networks
The code provided here uses the Brian simulator (http://briansimulator.org/). This code requires brian installation (http://brian2.readthedocs.io/en/stable/introduction/install.html). 

This code pertains to the recent Frontiers article:
'**Learning to Generate Sequences with Combination of Hebbian and Non-hebbian Plasticity in Recurrent Spiking Neural Networks- Priyadarshini Panda and Kaushik Roy' https://www.frontiersin.org/articles/10.3389/fnins.2017.00693/full**. 

This paper discusses an effective approach to generate simple words using a spiking reservoir (without output or readout neurons) using a combination of Hebbian STDP and non-Hebbian weight decay based learning of recurrent connections within the reservoir. The code provided here uses a similar spiking reservoir topology and combined plasticity learning rule to classify different characters (for character recognition using a simple synthesized training data of 800 examples of 7 different characters'C', 'R', 'O', 'T', 'F', 'A' taken from char74).

Please mention the full path to the directory (where the 'data/' folder resides) in line 46 in "recurrent_snn9_withdecay.py".

**Testing with pretrained weights:**
First run the main file "recurrent_snn9_withdecay.py" (which by default uses the pretrained weights/assignments from the weights folder) and wait until the simulation is finished to get the classification accuracy. 

**Training a new network:**
At first, run the "Reservoir_conn_generator.py". It ll randomly initialize the weights of the network and store in the random folder. Then run "recurrent_snn9_withdecay.py" by changing line 506 to "test_mode = False" to train the network.

**Note:**
In this simple demo the performance is evaluated using spiking rate of the excitatory neurons in the reservoir to make neuron assignments on the training set as described in https://www.frontiersin.org/articles/10.3389/fnins.2017.00693/full.

While we don't utilize the readout layer, the code does provide avenues to initialize and train the readout layer. You can do the same by changing line 507 to "tag_mode=True". However, as explained in the paper above, using STDP from input to reservoir and STDP + non-Hebbian decay for E-E connections within the reservoir is sufficient for training and classification. Thus, it is recommended to always keep tag_mode = False in both training and testing. 

While the current code utilizes a simple synthesized dataset, one can test it on MNIST and more complex recognition datasets.

**To cite this work, please use the following format:**
Panda, Priyadarshini, and Kaushik Roy. "Learning to generate sequences with combination of Hebbian and non-hebbian plasticity in recurrent spiking neural networks." Frontiers in neuroscience 11 (2017): 693.
