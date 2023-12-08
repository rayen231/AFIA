import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import re 
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS

def clean(text):
    cleaned_string = re.sub(r'[^A-Za-z\s]', '', text)
    return  cleaned_string
def clean_f(text):
    # Tokenize the input text
    tokens = word_tokenize(clean(text), language='french')

    # Get the list of French stop words
    french_stop_words = set(stopwords.words('french'))

    # Remove stop words from the tokens
    filtered_tokens = [word for word in tokens if word.lower() not in french_stop_words]

    # Join the filtered tokens into a string
    cleaned_text = ' '.join(filtered_tokens)

    return cleaned_text
def extract_nouns(tokens):
    # Join tokens into a space-separated string
    text = ' '.join(tokens)

    # Load the French language model
    nlp = spacy.load("fr_core_news_sm")

    # Process the text with spaCy
    doc = nlp(text)

    # Get nouns and filter out stop words
    nouns = [token.text.lower() for token in doc if token.pos_ == 'NOUN' and token.text.lower() not in STOP_WORDS]

    return nouns

def tokenize_text(text):
    # Use word_tokenize for tokenization
    tokens = word_tokenize(text)
    return extract_nouns(tokens)
# Specify the path to your JSON file
json_file_path = 'file.json'

# Open and read the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    # Load the JSON data into a Python list
    data = json.load(file)
l=[word for word in data['keywords']]
l1=[tokenize_text(i)for i in l]    
itents=[]
for i in l1:
    for j in i:
        itents.append(j)


def prin(ch):
    inputs_tokens =tokenize_text(ch)

    # Load pre-trained BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Function to get BERT embeddings for a given token
    def get_bert_embedding(token):
        inputs = tokenizer(token, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    total_similarity = 0
    num_pairs = 0

    for token1 in itents:
        for token2 in inputs_tokens:
            embedding1 = get_bert_embedding(token1)
            embedding2 = get_bert_embedding(token2)
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            total_similarity += similarity
            num_pairs += 1

    # Calculate the average similarity
    average_similarity = total_similarity / num_pairs

    # Compare the average similarity to the threshold (0.8)
    threshold = 0.8
    if average_similarity >= threshold:
        return True
    else:
        return False        