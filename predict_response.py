import nltk
import json
import pickle
import numpy as np
import random

import tensorflow
from data_preprocessing import get_stem_words

ignore_words = ['?', '!',',','.', "'s", "'m"]
model = tensorflow.keras.models.load_model('/chatbot_model.h5')

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)
words = pickle.load(open("./words.pkl", 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))

def preprocess_user_input(user_input):
  input_word_token1 = nltk.word_tokenize(user_input)
  input_word_token2 = get_stem_words(words=words, ignore_words=ignore_words)
  input_word_token2 = sorted(list(set(input_word_token2)))
  bag = []
  bag_of_words = []

  for word in words:
    if word in input_word_token2:
      bag_of_words.append(1)
    else:
      bag_of_words.append(0)
    
  bag.append(bag_of_words)

  return np.array(bag)

def bot_class_prediction(user_input):
  imp = preprocess_user_input(user_input)
  prediction = model.predict(imp)
  predicted_class_label = np.argmax(prediction[0])

  return predicted_class_label

def bot_response(user_input):
  predicted_class_label = bot_class_prediction(user_input)
  predicted_class = classes[predicted_class_label]

  for intent in intents:
    if intent['tag'] == predicted_class:
      bot_response = random.choice(intent['responses'])
  
  return bot_response

while True:
  user_input = input("Type your message here")
  print("User Input: ", user_input)
  response = bot_response(user_input)
  print("Bot Response: ", bot_response)

