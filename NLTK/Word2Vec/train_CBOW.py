# set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.create_vocabulary import tokenize_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


# read the text data from the disk
print("[INFO] reading the data from the disk...")
with open("data.txt") as filePointer:
  lines = filePointer.readlines()
textData = "".join(lines)
print(textData)

# tokenize the text data and store the vocabulary, the size of the tokenized text, and the tokenized text
(vocab, tokenizedTextSize, tokenizedText) = tokenize_data(data=textData)

# Map the vocab words to individual indices and map the indices to the words in vocab 
vocabToIndex = {
  uniqueWord:index for (index, uniqueWord) in enumerate(vocab)
}
indexToVocab = np.array(vocab)

# Convert the tokens into numbers 
textAsInt = np.array([vocabToIndex[word] for word in tokenizedText])

# Create the representational matrices as variable tensors 
contextVectorMatrix = tf.Variable(
  np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
  
)
centerVectorMatrix = tf.Variable(
  np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)