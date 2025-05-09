# Import the necessary packages
import tensorflow as tf
from tensorflow.keras.preprocessing.text import text_to_word_sequence

def tokenize_data(data):
    # Convert the data into token 
    tokenized_Text = text_to_word_sequence(input_text = data)

    # Create and store teh vocabulary of unique words along with the size of the tokenized text 
    vocab = sorted(set(tokenized_Text))
    tokenizedTextSize = len(tokenized_Text)
    
    

    # return the vocabulary, size of the tokenized text, and the tokenized text 
    return (vocab, tokenizedTextSize, tokenized_Text)