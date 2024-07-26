from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from scipy.spatial.distance import cosine
import os

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze()
    return vector.detach().numpy()

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def load_descriptions(directory):
    descriptions = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Assuming descriptions are in .txt files
            with open(os.path.join(directory, filename), 'r') as file:
                descriptions[filename] = file.read()
    return descriptions

def find_most_similar_description(prompt_vector, description_vectors):
    most_similar_description = None
    highest_similarity = -1
    for filename, vector in description_vectors.items():
        similarity = cosine_similarity(prompt_vector, vector)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_description = filename
    return most_similar_description

# Example directory containing description files
descriptions_directory = 'path/to/descriptions'
descriptions = load_descriptions(descriptions_directory)

# Convert descriptions to vectors
description_vectors = {filename: text_to_vector(text) for filename, text in descriptions.items()}

# Example prompt
prompt = "A happy dog playing in the park"
prompt_vector = text_to_vector(prompt)

# Find the most similar description
most_similar_description = find_most_similar_description(prompt_vector, description_vectors)
print(f"The most similar description is in file: {most_similar_description}")
