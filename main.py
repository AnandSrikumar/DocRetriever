from app.cli import get_args

from app.indexing import build_index
from app.search import Search

def main():
    args = get_args()

    if args.command == "index":
        build_index(args)
    
    elif args.command == "search":
        search = Search(args)
        while True:
            q = input("enter query: ")
            search.search(q)

if __name__ == "__main__":
    main()