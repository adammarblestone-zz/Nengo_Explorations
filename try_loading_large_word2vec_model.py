import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
import logging
import tarfile

def main():
    print "Loading model..."
    model = gensim.models.word2vec.Word2Vec.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary = True)
    model.init_sims(replace=True)
    vocab = model.index2word
    
    print vocab[:1000]
    
if __name__ == '__main__':
    main()