import gensim.downloader as api

model = api.load("word2vec-google-news-300")

model2 = api.load('fasttext-wiki-news-subwords-300')