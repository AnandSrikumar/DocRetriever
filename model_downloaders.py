# import gensim.downloader as api

# model = api.load("word2vec-google-news-300")

# model2 = api.load('fasttext-wiki-news-subwords-300')


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding = model.encode("Python is awesome")

print(embedding.shape)