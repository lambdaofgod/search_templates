from typing import Dict, List, Tuple

import requests
from pydantic import BaseModel, Field, validator


class SearchParams(BaseModel):
    top_k: int = Field(default=5)
    filters: dict = Field(default_factory=dict)


class SearchArgs(BaseModel):
    query: str
    params: SearchParams


class HaystackSearchClient(BaseModel):
    server_url: str

    @validator("server_url")
    def check_server_url(cls, v):
        try:
            docs_response = requests.get(f"{v}/docs")
            assert docs_response.status_code == 200, "Invalid server URL"
            return v
        except ConnectionError:
            raise ValueError(f"incorrect server url:\n{v}")

    def search(self, search_args: SearchArgs, displayed_meta_fields: List[str]):
        search_results = self._send_api_search_request(search_args)["documents"]
        return [
            self.make_displayable_result(result, displayed_meta_fields)
            for result in search_results
        ]

    def _send_api_search_request(self, search_args: SearchArgs):
        request_args = search_args.dict()
        response = requests.post(f"{self.server_url}/query", json=request_args)
        return response.json()

    @classmethod
    def make_displayable_result(cls, result, displayed_meta_fields: List[str]):
        result_meta = result["meta"]
        displayed_meta = {
            field: result_meta[field]
            for field in displayed_meta_fields
            if field in result_meta.keys()
        }
        return {**displayed_meta, "text": result["content"], "score": result["score"]}
