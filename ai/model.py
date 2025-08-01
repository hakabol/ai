import os
import json
import random
import importlib.util
import sys
import inspect

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

class chatbot_module(nn.Module):

    def __init__(self, inputsize, outputsize):
        super(chatbot_module, self).__init__()
        self.input_size = inputsize

        self.fc1 = nn.Linear(inputsize, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, outputsize)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class chatbot_assistance:

    def __init__(self):
        self.model = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.intents_path = os.path.join(current_dir, "intents.json")

        self.documents = []
        self.vocaluberries = []
        self.intents_responses = {}

        self.function_mapping =None

        self.X = None
        self.Y = None

    @staticmethod
    def token_lemon(text):
        lemonizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemonizer.lemmatize(word.lower()) for word in words]

        return words
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocaluberries]

    def pass_intents(self):
        lemonizer = nltk.WordNetLemmatizer()
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
            self.intents = []

            for intent in intents_data["intents"]:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent["responses"]
                
                for pattern in intent['patterns']:
                    pattern_words = self.token_lemon(pattern)
                    self.vocaluberries.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))
                
                self.vocaluberries = sorted(set(self.vocaluberries))

    def prepare_data(self):
        bags = []
        indecies = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indecies.append(intent_index)
        
        self.X = np.array(bags)
        self.Y = np.array(indecies)

    def load_settings(self, settings_path):
        with open(settings_path, "r") as f:
            self.intents_path = json.load(f)["intents_path"]
    
    def load(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.vocaluberries = []
        self.intents = []

        with open(self.intents_path, "r", encoding='utf-8') as f:
            file = json.load(f)
        for intents in file["intents"]:
            if intents['tag'] not in self.intents:
                    self.intents.append(intents['tag'])
                    self.intents_responses[intents['tag']] = intents["responses"]
            for intent in intents["patterns"]:
                pattern_word = self.token_lemon(intent)
                self.vocaluberries.extend(pattern_word)
                self.vocaluberries = sorted(set(self.vocaluberries))

        self.model = chatbot_module(dimensions["input_size"], dimensions["output_size"])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
    
    def process_message(self, input_message):
        words = self.token_lemon(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            
            predictions = self.model(bag_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)

        if confidence.item() < 0.00:
            return "Sorry, I didn't understand that."

        predicted_intent = self.intents[predicted_class_index]

        print("Predicted intent:", predicted_intent)
        print("Confidence:", confidence.item())
        print("Input:", input_message)
        print("Bag:", bag)

        if self.function_mapping:
            if predicted_intent in self.function_mapping:
                self.function_mapping[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        
        else:
            return None
