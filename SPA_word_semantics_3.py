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
    
    print "Loading Word2Vec model..."
    word2vec_model = gensim.models.Word2Vec.load("word2vec_model_1_cleaned")
    word2vec_model.init_sims(replace=True)
    word2vec_vocab = word2vec_model.index2word
    
    import readline
    readline.parse_and_bind("tab: complete")

    def complete(text,state):
       results = [x for x in word2vec_vocab if x.startswith(text)] + [None]
       return results[state]

    readline.set_completer(complete)

    print "This program uses an SPA network in Nengo to perform vector operations on a semantically structured word-vector space *learned* from a sentence corpus."
    print "When trained on a large corpus of English sentences, for example, it should produce: Vector[king] - Vector[man] + Vector[woman] = Vector[king]"
    
    print "For now, it just does subtraction..."
    print "\nPress <tab> twice to see all your autocomplete options."
    print "_______________________________________________________"
    line1 = raw_input('\nFirst word:> ')
    line2 = raw_input('\nSecond word:> ')

    if line1 and line2 in word2vec_vocab:
           val1 = word2vec_model[line1]
           val2 = word2vec_model[line2]
           diff = val1 - val2
           dot_products = [np.dot(word2vec_model[word2vec_model.index2word[i]], diff) for i in range(len(word2vec_model.index2word))]
           closest_word = word2vec_model.index2word[dot_products.index(max(dot_products))]
           print "\nWhat the Nengo model SHOULD return is something like: %s" % closest_word
    
    print "\nDefining SPA network..."
    model = spa.SPA(label = "Vector Storage")
    with model:
        
        # Dimensionality of each representation
        num_dimensions = 100
        sub_dimensions = 1
        
        # Create the vocabulary
        vocab = Vocabulary(num_dimensions, randomize = False)
                
        stored_value_1 = val1
        vocab.add("Stored_value_1", stored_value_1)
        
        stored_value_2 = val2
        vocab.add("Stored_value_2", stored_value_2)
        
        # Create a semantic pointer corresponding to the "correct" answer for the operation
        sum_vector = np.subtract(stored_value_1, stored_value_2)
        sum_vector = sum_vector/np.linalg.norm(sum_vector)
        vocab.add("Correct_target", sum_vector)

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
                
        # Control buffer
        model.control = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        control_probe = nengo.Probe(model.control.state.output)
        
        # Inputs to the word input buffers
        def first_input(t):
            return "Stored_value_1"
        def second_input(t):
            return "Stored_value_2"
        
        # Buffers to store the inputs:
        model.word_buffer1 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        model.word_buffer2 = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        
        # Probe to visualize the values stored in the buffers
        buffer_1_probe = nengo.Probe(model.word_buffer1.state.output)
        buffer_2_probe = nengo.Probe(model.word_buffer2.state.output)
        
        # Buffer to hold the result:
        model.result = spa.Buffer(dimensions = num_dimensions, subdimensions = sub_dimensions, neurons_per_dimension = 200, direct = True, vocab = vocab)
        result_probe = nengo.Probe(model.result.state.output)        
        
        # Control system        
        actions = spa.Actions('dot(control, Start_signal) --> result = word_buffer1 - word_buffer2', 'dot(control, Hold_signal) --> result = Hold_signal')
        model.bg = spa.BasalGanglia(actions)
        model.thalamus = spa.Thalamus(model.bg, subdim_channel = sub_dimensions)
        
        # Connect up the inputs
        model.input = spa.Input(control = control_input, word_buffer1 = first_input, word_buffer2 = second_input)
        
    # Start the simulator
    sim = nengo.Simulator(model)
    
    # Dynamic plotting
    plt.ion() # Dynamic updating of plots
    fig = plt.figure(figsize=(15,8))
    plt.show()
    ax = fig.gca()
    ax.set_title("Vector Storage")

    while True:
        sim.run(0.5) # Run for an additional 1 second
        plt.clf() # Clear the figure
        plt.plot(sim.trange(), similarity(sim.data, result_probe, vocab), label = "Buffer 3 Value")
        legend_symbols = vocab.keys
        plt.legend(legend_symbols, loc = 2)
        plt.draw() # Re-draw
        
        # Go back to our manually-stored vocabulary and see how well it did
        diff = sim.data[result_probe][-1]
        dot_products = [np.dot(word2vec_model[word2vec_model.index2word[i]], diff) for i in range(len(word2vec_model.index2word))]
        closest_word = word2vec_model.index2word[dot_products.index(max(dot_products))]
        print "Time: %f" % sim.trange()[-1]
        print "\nWhat the Nengo model DID return is something like: %s" % closest_word
        print "\n"
        
    plt.show()
    
if __name__ == '__main__':
    main()