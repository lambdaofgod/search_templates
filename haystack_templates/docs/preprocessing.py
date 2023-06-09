import pandas as pd
from typing import Callable, List
import abc
from pydantic import BaseModel
from pandera.typing import Index, DataFrame, Series
import pandera as pa


class InputSchema(pa.DataFrameModel):
    chapter_id: Series[str]
    chapter_title: Series[str]
    sections: Series[List[dict]]
    text: Series[str] = pa.Field(nullable=True)


class BookSchema(pa.DataFrameModel):
    chapter_id: Series[str]
    chapter_title: Series[str]
    chapter_text: Series[str] = pa.Field(nullable=True)
    section_text: Series[str] = pa.Field(nullable=True)
    section: Series[str] = pa.Field(nullable=True)
    subsection: Series[str] = pa.Field(nullable=True)
    subsection_text: Series[str] = pa.Field(nullable=True)


InputDFType = DataFrame[InputSchema]
BookDFType = DataFrame[BookSchema]


class Preprocessor(abc.ABC):
    @abc.abstractmethod
    def preprocess_df(self, df: InputDFType) -> BookDFType:
        pass


class CoercionPreprocessor(Preprocessor):
    def preprocess_df(self, df: InputDFType) -> BookDFType:
        return df


class CombinedPreprocessor(Preprocessor, BaseModel):
    positive_preprocessor: Preprocessor
    negative_preprocessor: Preprocessor
    condition: Callable[[pd.DataFrame], pd.Series]

    def preprocess_df(self, df: InputDFType) -> BookDFType:
        positive_df = df[self.condition(df)]
        negative_df = df[~self.condition(df)]
        positive_df = self.positive_preprocessor.preprocess_df(positive_df)
        negative_df = self.negative_preprocessor.preprocess_df(negative_df)
        return positive_df.append(negative_df).reset_index(drop=True)

    class Config:
        arbitrary_types_allowed = True


class SectionSchema(pa.DataFrameModel):
    section_text: Series[str]
    section: Series[str]
    chapter_id: Series[str]


class SectionPreprocessor(Preprocessor):
    @pa.check_types
    def preprocess_df(self, df: InputDFType) -> BookDFType:
        sections_df = self._get_nested_df(df)
        processed_df = df.merge(sections_df, on="chapter_id").rename(
            {"text": "chapter_text"}, axis=1
        )

        return processed_df.drop(columns=["sections"])

    def _get_nested_df(self, df):
        sections_df = self._get_nested_level_df(df, "sections", ["chapter_id"])
        print(sections_df.head())
        subsections_df = self._get_nested_level_df(
            sections_df, "subsections", ["sections", "chapter_id"]
        )
        return (
            sections_df.drop(columns=["subsections"])
            .merge(subsections_df, on=["chapter_id", "sections"])
            .rename(
                lambda col_name: col_name.replace("sections", "section"),
                axis=1,
            )
        )

    def _get_nested_level_df(self, df, level_name, prev_levels):
        exploded_df = df.explode(level_name).reset_index()
        sections_df = pd.json_normalize(
            exploded_df[level_name],
        ).rename(columns={"text": f"{level_name}_text", "title": level_name})
        return sections_df.assign(
            **{
                prev_level_name: exploded_df[prev_level_name]
                for prev_level_name in prev_levels
            }
        )
