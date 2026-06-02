from app.cli import get_args

from app.indexing import build_index

def main():
    args = get_args()

    if args.command == "index":
        build_index(args)
    
    elif args.command == "search":
        print("search")

if __name__ == "__main__":
    main()