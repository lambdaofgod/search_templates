import json
import logging
import pprint
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pandera as pa
import requests
import tqdm
from fastapi import Depends, File, Form, UploadFile
from pandera import Check, Column, DataFrameSchema
from pydantic import BaseModel, Field, validator
from requests.exceptions import ConnectionError
from tqdm.contrib.concurrent import process_map

logging.basicConfig(level="INFO")


class DataframeIndexerConfig(BaseModel):
    text_col: str
    metadata_cols: List[str]
    haystack_server_url: str
    max_workers: int = Field(default=10)

    @property
    def input_df_schema(self):
        return DataFrameSchema(
            {
                self.text_col: Column(str),
                **{col: Column(str, nullable=True) for col in self.metadata_cols},
            }
        )


class HaystackClient(BaseModel):
    server_url: str

    @validator("server_url")
    def check_server_url(cls, v):
        try:
            docs_response = requests.get(f"{v}/docs")
            assert docs_response.status_code == 200, "Invalid server URL"
            return v
        except ConnectionError:
            raise ValueError(f"incorrect server url:\n{v}")

    def upload_files(self, paths: List[str], metadata: List[dict], verbose=True):
        assert len(paths) == len(metadata)
        # this is pretty silly as it would be better to actually upload batches of files
        paths_meta_iter = zip(paths, metadata)
        if verbose:
            paths_meta_iter = tqdm.tqdm(paths_meta_iter, total=len(paths))
        for p, meta in paths_meta_iter:
            self._send_request_f(Path(p), meta, verbose)

    def _send_request_f(self, file_path, meta, verbose):
        with file_path.open("rb") as f:
            response = self._send_request(file_path, f, meta=meta)
            if verbose:
                if response.status_code != 200:
                    logging.info(f"Error uploading file: {file_path.name}")
                    logging.info(response.reason)
                else:
                    logging.info(f"Successfully uploaded file: {file_path.name}")

    def _send_request(
        self,
        file_path,
        f,
        meta: dict = {},
        additional_params: Optional[str] = None,
    ):
        headers = {"accept": "application/json", "Content-Type": "multipart/form-data"}
        url = f"{self.server_url}/file-upload"
        data = {
            "split_overlap": "",
            "meta": json.dumps(meta),
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


class DataframeIndexer(BaseModel):
    config: DataframeIndexerConfig
    client: HaystackClient

    @classmethod
    def make_from_config(cls, config: DataframeIndexerConfig):
        client = HaystackClient(server_url=config.haystack_server_url)
        return DataframeIndexer(config=config, client=client)

    def index_df(self, df):
        self._validate_input_df_schema(df)
        with tempfile.TemporaryDirectory() as dirpath:
            df["path"] = df[self.config.text_col].apply(
                lambda text: f"{dirpath}/{self._get_tmp_file_name(text)}.txt"
            )
            self.store_df_texts(df)

    @classmethod
    def _get_tmp_file_name(cls, text):
        return str(hash(text)).replace("\/", "")

    def store_df_texts(self, df: pd.DataFrame):
        indexed_rows = process_map(
            self._index_row_dict,
            [row.to_dict() for (_, row) in df.iterrows()],
            max_workers=self.config.max_workers,
            chunksize=1,
        )
        for row in indexed_rows:
            logging.info(f"stored {row}")

    def _index_row_dict(self, row_dict):
        meta = {
            metadata_col: row_dict[metadata_col]
            for metadata_col in self.config.metadata_cols
            if row_dict[metadata_col] is not None
        }
        text = row_dict[self.config.text_col]
        path = row_dict["path"]
        self._index_text(text, meta=meta, path=path)

    def _index_text(self, text, meta, path):
        with open(path, "w") as f:
            f.write(text)
        self.client.upload_files([path], [meta], verbose=False)

    def _validate_input_df_schema(self, df):
        self.config.input_df_schema.validate(df)
