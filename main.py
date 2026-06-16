import json
from uuid import uuid4

from app.cli import get_args

from app.indexing import build_index
# from app.searching.search import Search

def save_query_res(result):
    with open(f"results/{uuid4()}.json", 'w') as fp:
        json.dump(result, fp, indent=2)

def main():
    args = get_args()

    if args.command == "index":
        build_index(args)
    
    # elif args.command == "search":
    #     search = Search(args)
    #     while True:
    #         try:
    #             q = input("enter query: ")
    #             result = search.search(q)
    #             save_query_res(result)
    #             for res in result:
    #                 res.pop("text")
    #                 print(res)
    #         except KeyboardInterrupt:
    #             break
    #         except Exception as e:
    #             print("try again...", e)
    #             continue

if __name__ == "__main__":
    main()