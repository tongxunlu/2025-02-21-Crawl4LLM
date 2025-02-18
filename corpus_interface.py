import gzip
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


class DocumentAnnotation(dict):
    _compare_key: str | None = None
    _order: str = "desc"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def set_compare_method(cls, key: str, order: Literal["desc", "asc"]) -> None:
        cls._compare_key = key
        if order not in ["desc", "asc"]:
            raise ValueError("Order must be either 'desc' (descending) or 'asc' (ascending)")
        cls._order = order

    @classmethod
    def get_compare_key(cls) -> str | None:
        return cls._compare_key

    def __lt__(self, other) -> bool:
        if self._compare_key is None:
            raise ValueError("Compare key not set")
        if self._order == "desc":
            return self[self._compare_key] > other[self._compare_key]
        return self[self._compare_key] < other[self._compare_key]


@dataclass
class Document:
    docid: str
    text: str | None = None
    annotations: DocumentAnnotation = field(default_factory=DocumentAnnotation)


class ClueWeb22Api:
    # Modified from https://github.com/lemurproject/ClueWeb22/blob/main/ClueWeb22Api.py
    def __init__(self, cw22root_path) -> None:
        self.cw22root_path = cw22root_path

    def get_base_filename_by_id(self, cw22id: str, file_type: str = "html") -> str:
        html_path = self.cw22root_path + os.sep + file_type
        id_parts = cw22id.split("-")

        language = id_parts[1][:2]
        segment = id_parts[1][:4]
        directory = id_parts[1]
        base_path = html_path + os.sep + language + os.sep + segment + os.sep + directory + os.sep
        base_filename = base_path + id_parts[1] + "-" + id_parts[2]
        return base_filename

    def get_json_record(self, cw22id: str, record_type: str) -> str:
        base_filename = self.get_base_filename_by_id(cw22id, file_type=record_type)

        id_parts = cw22id.split("-")
        doc = int(id_parts[len(id_parts) - 1])

        offset_length = len("{:010d}\n".format(0, 0))
        offset_path = base_filename + ".offset"
        json_path = base_filename + ".json.gz"
        with open(json_path, "rb") as f_json:
            with open(offset_path, "r") as f_offset:
                f_offset.seek(int(doc) * int(offset_length))
                start_bytes = int(f_offset.read(offset_length).strip())
                end_bytes = int(f_offset.read(offset_length).strip())
                f_json.seek(start_bytes)
                record = f_json.read(end_bytes - start_bytes)
                record = gzip.decompress(record).decode("utf-8")
                return record

    def get_clean_text(self, cw22id: str) -> str:
        record = self.get_json_record(cw22id, "txt")
        return record

    def get_inlinks(self, cw22id: str) -> str:
        record = self.get_json_record(cw22id, "inlink")
        return record

    def get_outlinks(self, cw22id: str) -> str:
        record = self.get_json_record(cw22id, "outlink")
        return record


class UnifiedGetter:
    def __init__(self, cw22_api: ClueWeb22Api, docid_pos: int = 0) -> None:
        self.cw22_api = cw22_api
        self.docid_pos = docid_pos

    def get_doc(self, docid: str) -> Document | None:
        try:
            cw22_data = json.loads(self.cw22_api.get_clean_text(docid))
        except:
            logger.debug(f"Failed to get doc: {docid}")  # Too many documents not found
            return None
        assert cw22_data["ClueWeb22-ID"] == docid
        return Document(docid=docid, text=cw22_data["Clean-Text"])

    def get_outlinks(self, docid: str) -> list[str]:
        try:
            obj = json.loads(self.cw22_api.get_outlinks(docid))
        except:  # File not found or empty entry
            logger.info(f"Failed to get outlinks for doc: {docid}")
            return []
        assert obj["ClueWeb22-ID"] == docid
        return [
            x[self.docid_pos]
            for x in obj["outlinks"]
            if x[self.docid_pos] is not None
            and x[self.docid_pos].startswith(f"clueweb22-en0")  # Only keep CW22-A outlinks
        ]

    def get_inlinks(self, docid: str) -> list[str]:
        try:
            obj = json.loads(self.cw22_api.get_inlinks(docid))
        except:
            logger.debug(f"Failed to get inlinks for doc: {docid}")
            return []
        assert obj["ClueWeb22-ID"] == docid
        return [
            x[self.docid_pos]
            for x in obj["anchors"]
            if x[self.docid_pos] is not None
            and x[self.docid_pos].startswith(f"clueweb22-en0")  # Only keep CW22-A inlinks
        ]
