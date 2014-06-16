import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
import logging

import nengo
import nengo.spa as spa
from nengo.spa import Vocabulary
from nengo.spa.utils import similarity


'''
This program implements a spiking neural network that performs vector operations on a semantically-organized word-vector space derived from deep learning on a document corpus.

It should be able to perform operations like the canonical example: king - man + woman = queen.

It is based on the Nengo SPA cheat sheet demo from Terry Stewart et al.

It runs in native Python from the terminal, rather than using the online GUI.
'''

def main():
    
    model = spa.SPA(label="Vector Storage")
    with model:
        
        # Dimensionality of each representation
        num_dimensions = 1
        sub_dimensions = 1
        
        # Create the vocabulary
        vocab = Vocabulary(num_dimensions, randomize = False)
        
        # Form the inputs
        stored_value_1 = [0.6/np.sqrt((0.6**2) * num_dimensions)] * num_dimensions
        vocab.add("Stored_value_1", stored_value_1)
        
        stored_value_2 = [(-1**i)*0.6/np.sqrt((0.6**2) * num_dimensions) for i in range(num_dimensions)]
        vocab.add("Stored_value_2", stored_value_2)
                
        def first_input(t):
            if t < 10:
                return "Stored_value_2"
            else:
                return "Stored_value_1"
        
        def second_input(t):
            if t < 5:
                return "Stored_value_1"
            else:
                return "Stored_value_2"
                
        # Buffers to store the input
        model.buffer1 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 150)
        model.buffer2 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 150)
        
        # Probe to visualize the values stored in the buffers
        buffer_1_probe = nengo.Probe(model.buffer1.state.output)
        buffer_2_probe = nengo.Probe(model.buffer2.state.output)
        
        # Connect up the inputs
        model.input = spa.Input(buffer1 = first_input)
        model.input = spa.Input(buffer2 = second_input)
        

        # Buffer to store the output
        model.buffer3 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 150)
        buffer_3_probe = nengo.Probe(model.buffer3.state.output)
        
        # Control system
        actions = spa.Actions('dot(buffer1, Stored_value_2) --> buffer3=Stored_value_2', 'dot(buffer1, Stored_value_1) --> buffer3=Stored_value_1-Stored_value_2')
        model.bg = spa.BasalGanglia(actions)
        model.thalamus = spa.Thalamus(model.bg)

        
    # Start the simulator
    sim = nengo.Simulator(model)

    # Dynamic plotting
    plt.ion() # Dynamic updating of plots
    fig = plt.figure(figsize=(15,8))
    plt.show()
    ax = fig.gca()
    ax.set_title("Vector Storage")

    while True:
        sim.run(1) # Run for an additional 1 second
        plt.clf() # Clear the figure
        plt.plot(sim.trange(), similarity(sim.data, buffer_1_probe, vocab), label = "Buffer 1 Value") # Plot the entire dataset so far
        plt.plot(sim.trange(), similarity(sim.data, buffer_2_probe, vocab), label = "Buffer 2 Value")
        plt.plot(sim.trange(), similarity(sim.data, buffer_3_probe, vocab), label = "Buffer 3 Value")
        plt.legend(vocab.keys * 3, loc = 2)
        plt.draw() # Re-draw
    
if __name__ == '__main__':
    main()