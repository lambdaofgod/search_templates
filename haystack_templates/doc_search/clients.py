import requests
from typing import Optional, List
from pathlib import Path
from fastapi import File, Form, Depends, UploadFile
from pydantic import BaseModel, validator
from requests.exceptions import ConnectionError
import logging
import json
import pprint
import tqdm.contrib.concurrent


logging.basicConfig(level="INFO")


class DocSearchClient(BaseModel):
    server_url: str

    @validator("server_url")
    def check_server_url(cls, v):
        try:
            docs_response = requests.get(f"{v}/docs")
            assert docs_response.status_code == 200, "Invalid server URL"
            return v
        except ConnectionError:
            raise ValueError(f"incorrect server url:\n{v}")

    def upload_files(self, directory: str, glob_path: str):
        directory_path = Path(directory)
        file_paths = [path for path in directory_path.glob(glob_path) if path.is_file()]

        # this is pretty silly as it would be better to actually upload batches of files
        tqdm.contrib.concurrent.process_map(
            self._send_request_f, file_paths, max_workers=4
        )

    def _send_request_f(self, file_path):
        with file_path.open("rb") as f:
            response = self._send_request(file_path, f)
            if response.status_code != 200:
                logging.info(f"Error uploading file: {file_path.name}")
                logging.info(response.reason)
            else:
                logging.info(f"Successfully uploaded file: {file_path.name}")

    def _send_request(
        self,
        file_path,
        f,
        meta: Optional[str] = None,
        additional_params: Optional[str] = None,
    ):
        headers = {"accept": "application/json", "Content-Type": "multipart/form-data"}
        url = f"{self.server_url}/file-upload"
        data = {
            "split_overlap": "",
            "meta": "null",
            "split_respect_sentence_boundary": "",
            "split_length": "",
            "remove_numeric_tables": "",
            "clean_whitespace": "",
            "clean_header_footer": "",
            "clean_empty_lines": "",
            "valid_languages": "",
            "split_by": "",
            "additional_params": "null",
        }
        files = {"files": (file_path.name, f, "text/plain")}

        # Send the file to the given URL as a POST request with the prepared data
        response = requests.post(url, data=data, files=files)
        return response
