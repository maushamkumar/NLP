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
  uniqueWord: index for (index, uniqueWord) in enumerate(vocab)
}
indexToVocab = np.array(vocab)

# Convert the tokens into numbers 
textAsInt = np.array([vocabToIndex[word] for word in tokenizedText])

# Create the representational matrices as variable tensors 
contextVectorMatrix = tf.Variable(
  np.random.rand(len(vocab), config.EMBEDDING_SIZE)
)
centerVectorMatrix = tf.Variable(
  np.random.rand(len(vocab), config.EMBEDDING_SIZE)
)

# Initialize the optimizer and create an empty list to log the loss 
optimizer = tf.keras.optimizers.Adam()
lossList = list()

# loop over the training epochs 
print("[INFO] Starting CBOW training")
for epoch in tqdm(range(config.NUM_ITERATIONS)):
  # initialize the loss per epoch 
  lossPerEpoch = 0 
  
  # the window for center vector prediction is created 
  for start in range(tokenizedTextSize - config.WINDOW_SIZE):
    # generate the indices for the window 
    indices = textAsInt[start: start + config.WINDOW_SIZE]
    
    # initialize the gradient tape 
    with tf.GradientTape() as tape:
      # initialize the context vector 
      combinedContext = 0.0
      
      # loop over the indices and grab the neighboring 
      # word representation from the embedding matrix 
      for count, index in enumerate(indices):
        if count != config.WINDOW_SIZE // 2:
          combinedContext += contextVectorMatrix[index, :]
      
      # standardize the result according to the window size 
      combinedContext /= (config.WINDOW_SIZE - 1)
      
      # calculate the center word embedding prediction 
      output = tf.matmul(centerVectorMatrix, tf.expand_dims(combinedContext, 1))
      
      # apply softmax to get probabilities 
      softOut = tf.nn.softmax(output, axis=0)
      
      # get the index of the center word 
      centerIndex = indices[config.WINDOW_SIZE // 2]
      
      # calculate the loss for the center word 
      loss = softOut[centerIndex]
      
      # calculate the logarithmic loss 
      logLoss = -tf.math.log(loss + 1e-10)  # added epsilon for numerical stability
      
    # Update the loss per epoch and apply 
    # the gradients to the embedding matrices 
    lossPerEpoch += logLoss.numpy()
    grad = tape.gradient(
      logLoss, [contextVectorMatrix, centerVectorMatrix]
    )
    optimizer.apply_gradients(
      zip(grad, [contextVectorMatrix, centerVectorMatrix])
    )
    
  # Update the loss list 
  lossList.append(lossPerEpoch)


# create output directory if it doesn't already exist 
if not os.path.exists(config.OUTPUT_PATH):
  os.makedirs(config.OUTPUT_PATH)
  
# plot the loss for evaluation 
print('[INFO] plotting loss .. ')
plt.plot(lossList)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.CBOW_LOSS)


# apply dimenional reductionality using tsne for the representation matrices 
tsneEmbed = (
  TSNE(n_components=2).fit_transform(centerVectorMatrix.numpy())
)

tsbeDecode = (
  TSNE(n_components=2).fit_transform(centerVectorMatrix.numpy())
)

# initialize a index counter 
indexCount = 0 

# initialize the tsne figure 
plt.figure(figsize=(25, 5))

# loop over the tsne embeddings and plot the corresponding words 
print("[INFO] plotting TSNE embeddings ... ")
for (word, embedding) in tsbeDecode[:100]:
  plt.scatter(word, embedding)
  plt.annotate(indexToVocab[indexCount], (word, embedding))
  indexCount += 1 
plt.savefig(config.CBOW_TSNE)