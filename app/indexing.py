from argparse import Namespace
import time

from app.chunker import chunk_docs
from app.loader import create_doc_id_map, load_docs
from app.preprocess import preprocess_text


def build_index(args: Namespace):
    docs = load_docs(args.data_dir)
    doc_id_map = create_doc_id_map(docs)
    print("chunking...")
    s = time.perf_counter()
    chunks = chunk_docs(docs, args.chunking, args.chunk_size, args.chunk_overlap)
    e = time.perf_counter()
    print(f"chunking done.....\n total: {len(chunks)}\ntook: {e-s} secs")
    cleaned_chunks = preprocess_text(chunks, True)
    print(f"cleaned chunks....\ntook:{time.perf_counter() - e} secs")
    print(f"a sample chunk: {type(cleaned_chunks[0])} --> {cleaned_chunks[0][0: 10]}")
