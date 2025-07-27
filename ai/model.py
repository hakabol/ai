import os
import json
import random
import inspect
import importlib.util

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

class chatbot_module(nn.Module):
    def __init__(self, input_size, output_size):
        super(chatbot_module, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
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
        self.intents_path = None
        self.function_mapping = None
        self.documents = []
        self.vocab = []
        self.intents = []
        self.intents_responses = {}
        self.X = None
        self.Y = None

    @staticmethod
    def token_lemon(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words):
        return [1 if w in words else 0 for w in self.vocab]

    def pass_intents(self):
        if not self.intents_path or not os.path.exists(self.intents_path):
            raise FileNotFoundError("intents_path not set or file not found.")
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        for intent in intents_data["intents"]:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent["responses"]
            for pattern in intent['patterns']:
                pattern_words = self.token_lemon(pattern)
                self.documents.append((pattern_words, tag))
                self.vocab.extend(pattern_words)
        self.vocab = sorted(set(self.vocab))

    def prepare_data(self):
        bags, indices = [], []
        for words, tag in self.documents:
            bags.append(self.bag_of_words(words))
            indices.append(self.intents.index(tag))
        self.X = np.array(bags)
        self.Y = np.array(indices)

    def train(self, batch_size=8, lr=0.001, epochs=1000):
        if self.X.size == 0 or self.Y.size == 0:
            raise ValueError("Training data is empty.")
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = chatbot_module(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.6f}")

    def save(self, model_path, dimensions_path, settings_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({"input_size": self.X.shape[1], "output_size": len(self.intents)}, f)
        with open(settings_path, 'w') as f:
            f.write(f"intents_path = '{self.intents_path}'\n\n")
            for func in self.function_mapping.values():
                f.write(f"{inspect.getsource(func)}\n\n")
            f.write("function_mapping = {\n")
            for tag, func in self.function_mapping.items():
                f.write(f"    \"{tag}\": {func.__name__},\n")
            f.write("}\n")

    def load_settings(self, settings_path):
        spec = importlib.util.spec_from_file_location("settings", settings_path)
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        self.intents_path = settings.intents_path
        self.function_mapping = settings.function_mapping

    def load(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dims = json.load(f)
        self.model = chatbot_module(dims["input_size"], dims["output_size"])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message):
        words = self.token_lemon(input_message)
        bag = self.bag_of_words(words)
        if len(bag) != self.model.fc1.in_features:
            return "Input shape mismatch. Model expects different input size."
        input_tensor = torch.tensor([bag], dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, dim=1)

        if confidence.item() < 0.6:
            return "Sorry, I didn't understand that."

        intent = self.intents[predicted_idx.item()]
        if self.function_mapping and intent in self.function_mapping:
            result = self.function_mapping[intent]()
            if result:
                return result
        return random.choice(self.intents_responses.get(intent, ["..."]))
