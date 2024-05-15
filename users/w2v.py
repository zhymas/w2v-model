from gensim.models import Word2Vec
from .models import Record
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

nltk.download('punkt')

def train_model():
    records = Record.objects.all()
    
    corpus = [word_tokenize(record.content.lower()) for record in records]
    
    print(f"Corpus: {corpus}")
    if not corpus:
        print("Corpus is empty.")
        return
    
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
    
    model.build_vocab(corpus)
    print("Vocabulary built successfully.")
    
    vocab_size = len(model.wv)
    print(f"Vocabulary size: {vocab_size}")
    if vocab_size == 0:
        print("Vocabulary is empty.")
        return
    
    model.train(corpus, total_examples=model.corpus_count, epochs=10)
    print("Model trained successfully.")
    
    model.save("word2vec.model")
    print("Model saved successfully.")

def generate_embeddings():
    model = Word2Vec.load("word2vec.model")
    
    records = Record.objects.all()

    for record in records:
        tokens = word_tokenize(record.content.lower())
        embedding = [model.wv[token] for token in tokens if token in model.wv]
        if embedding:
            avg_embedding = np.mean(embedding, axis=0)
            record.set_embedding(avg_embedding.tolist())
            record.save()
