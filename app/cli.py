from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="CLI for DocRetriever")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="Create document index")
    index_parser.add_argument("--data-dir", required=True)
    index_parser.add_argument("--chunking", default="recursive")
    index_parser.add_argument("--chunk-size", default=1000)
    index_parser.add_argument("--chunk-overlap", default=100)

    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--query", required=True)
    return parser.parse_args()
