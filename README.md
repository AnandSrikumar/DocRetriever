# DocRetriever
A light weight search engine for documents

# how to use

1. Indexing --> creates pickles in designated folder, these are the vectors and embeddings of the documents that will be used during searching
2. search --> loads pickes and then takes input for query, searches relevant documents

## supported doc types

1. docx
2. pdf
3. txt

#### future additions

1. md
2. html


## Indexing

```python
index_parser = subparsers.add_parser("index", help="Create document index")
index_parser.add_argument("--data-dir", required=True)
index_parser.add_argument("--chunking", default="recursive")
index_parser.add_argument("--chunk-size", default=1000)
index_parser.add_argument("--chunk-overlap", default=100)
index_parser.add_argument("--index-loc", default="indexed/")
```

1. --data-dir --> this is required field where you store all the documents that needs querying

```
python main.py index --data-dir data/
```

remaining fields have default values.

this step will create `.pkl` files in the `indexed/` folder

## searching

```
search_parser = subparsers.add_parser("search", help="Search documents")
search_parser.add_argument("--search-method", default="tfidf")
search_parser.add_argument("--index-loc", default="indexed/")
search_parser.add_argument("--top-k", default=3)
```

here, all fields are optional

### supported search methods

1. tfidf
2. bow
3. word2vec
4. fasttext

#### future
1. bm25
2. sentence-transformers

```
python main.py search --search-method tfidf  --top-k 1

python main.py search --search-method word2vec  --top-k 4
```

## performance

<b> TFIDF seems to perform the best </b>

## running it

check main.py, you can save the results in a json too. feel free to modify

# Optimizations needed

1. The loading of embedding model is too slow, need to investigate
2. speed up the search for tfidf and bow
3. speed up the indexing. explore parallelism for embeddings