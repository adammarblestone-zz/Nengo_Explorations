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
from nengo.spa.assoc_mem import AssociativeMemory


'''
This program implements a spiking neural network that performs vector operations on a semantically-organized word-vector space derived from deep learning on a document corpus.

It should be able to perform operations like the canonical Word2Vec example: king - man + woman = queen.
'''

def main():
    
    model = spa.SPA(label = "Vector Storage")
    with model:
        
        # Dimensionality of each representation
        num_dimensions = 30
        sub_dimensions = 1
        
        # Create the vocabulary
        vocab = Vocabulary(num_dimensions, randomize = False)
        
        # Form the inputs manually by directly defining their vectors
        stored_value_1 = np.random.rand(num_dimensions) - [0.5] * num_dimensions
        stored_value_1 = [s/np.linalg.norm(stored_value_1) for s in stored_value_1]
        vocab.add("Stored_value_1", stored_value_1)
        
        stored_value_2 = np.random.rand(num_dimensions) - [0.5] * num_dimensions
        stored_value_2 = [s/np.linalg.norm(stored_value_2) for s in stored_value_2]
        vocab.add("Stored_value_2", stored_value_2)
        
        stored_value_3 = np.random.rand(num_dimensions) - [0.5] * num_dimensions
        stored_value_3 = [s/np.linalg.norm(stored_value_3) for s in stored_value_3]
        vocab.add("Stored_value_3", stored_value_3)
        
        # Create a semantic pointer corresponding to the "correct" answer for the operation
        sum_vector = np.subtract(np.add(stored_value_1, stored_value_2), stored_value_3)
        sum_vector = sum_vector/np.linalg.norm(sum_vector)
        vocab.add("Sum", sum_vector)

        # Define the control signal inputs as random vectors
        r1 = [1] * num_dimensions
        r1 = r1 / np.linalg.norm(r1)
        r2 = [(-1)**k for k in range(num_dimensions)]
        r2 = r2 / np.linalg.norm(r2)
        vocab.add("Hold_signal", r1)
        vocab.add("Start_signal", r2)

        # Control when the vector operation takes place
        def control_input(t):
            if t < 1:
                return "Hold_signal"
            else:
                return "Start_signal"
        
        # inputs to the input word buffers
        def first_input(t):
            return "Stored_value_1"
        def second_input(t):
            return "Stored_value_2"
        def third_input(t):
            return "Stored_value_3"
        
        # Control buffer
        model.control = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        control_probe = nengo.Probe(model.control.state.output)
        

        # Buffers to store the inputs: e.g., King, Man, Woman
        model.word_buffer1 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        model.word_buffer2 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        model.word_buffer3 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        
        # Probe to visualize the values stored in the buffers
        buffer_1_probe = nengo.Probe(model.word_buffer1.state.output)
        buffer_2_probe = nengo.Probe(model.word_buffer2.state.output)
        buffer_3_probe = nengo.Probe(model.word_buffer3.state.output)
        
        # Buffer to hold the result: e.g. Queen
        model.result = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        result_probe = nengo.Probe(model.result.state.output)        
        
        # Control system        
        actions = spa.Actions('dot(control, Start_signal) --> result = word_buffer1 + word_buffer2 - word_buffer3')
        model.bg = spa.BasalGanglia(actions)
        model.thalamus = spa.Thalamus(model.bg, subdim_channel = sub_dimensions)
        
        # Connect up the inputs
        model.input = spa.Input(control = control_input, word_buffer1 = first_input, word_buffer2 = second_input, word_buffer3 = third_input)
        
        '''
        # Cleanup the output
        n_cleanup_neurons = 20 # number of neurons per item in vocabulary
        model.associative_memory = AssociativeMemory(input_vocab = vocab, n_neurons_per_ensemble = n_cleanup_neurons, threshold = 0.3)

        model.cleanup_result = spa.Buffer(dimensions = num_dimensions, subdimensions = 1, neurons_per_dimension = 50, vocab = vocab)

        # Connect up the cleanup memory
        nengo.Connection(model.result.state.output, model.associative_memory.input)
        nengo.Connection(model.associative_memory.output, model.cleanup_result.state.input)

        # And probe it
        cleanup_probe = nengo.Probe(model.cleanup_result.state.output)
        '''
        
    # Start the simulator
    sim = nengo.Simulator(model)
    sim.run(4) # Run for an additional 1 second
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.gca()
    ax.set_title("Control Input")
    plt.plot(sim.trange(), np.abs(similarity(sim.data, control_probe, vocab)), label = "Control value")
    plt.legend(vocab.keys * 3, loc = 2)
    plt.ylim(-1.1, 1.1)
    
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.gca()
    ax.set_title("Result")
    plt.plot(sim.trange(), similarity(sim.data, result_probe, vocab), label = "Result value")
    plt.legend(vocab.keys * 3, loc = 2)
        
    plt.show()
    
if __name__ == '__main__':
    main()