import fire
from doc_search.clients import DocSearchClient


class Main:
    @staticmethod
    def index(server_url, directory, glob_path):
        client = DocSearchClient(server_url=server_url)
        client.upload_files(directory=directory, glob_path=glob_path)


if __name__ == "__main__":
    fire.Fire(Main())
