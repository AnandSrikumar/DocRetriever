from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="CLI for DocRetriever")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Create document index")
    index_parser.add_argument("--data-dir", required=True)
    index_parser.add_argument("--chunking", default="recursive")
    index_parser.add_argument("--chunk-size", default=1000)
    index_parser.add_argument("--chunk-overlap", default=100)
    index_parser.add_argument("--index-loc", default="indexed/")

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--search-method", default="tfidf")
    search_parser.add_argument("--index-loc", default="indexed/")
    search_parser.add_argument("--top-k", default=3)
    return parser.parse_args()
