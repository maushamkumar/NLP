# Import the necessary packages
import os 

# Define the number of embedding dimensions 
EMBEDDING_SIZE = 10

# Define the window size and number of iterations 
WINDOW_SIZE = 2
NUM_ITERATIONS = 1000

# Define the path to the output directory
OUTPUT_PATH = 'output'

# Define the path to the skipgram outputs
SKIPGRAM_LOSS = os.path.join(OUTPUT_PATH, 'skipgram_loss.png')
SKIPGRAM_TSNE = os.path.join(OUTPUT_PATH,'tsne_skipgram.png')

# Define the path to the CBOW outputs
CBOW_LOSS = os.path.join(OUTPUT_PATH,'cbow_loss.png')
CBOW_TSNE = os.path.join(OUTPUT_PATH,'tsne_cbow.png')