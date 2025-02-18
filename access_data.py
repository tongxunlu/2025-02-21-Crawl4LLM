import logging
from corpus_interface import ClueWeb22Api, UnifiedGetter

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    cw22 = UnifiedGetter(
        ClueWeb22Api("/bos/tmp6/ClueWeb22_A"), docid_pos=5
    )  # path to ClueWeb22_A on Boston
    doc_content = cw22.get_doc("clueweb22-en0045-44-19547")
    print(doc_content)
    outlinks = cw22.get_outlinks("clueweb22-en0045-44-19547")
    print(outlinks)
    for outlink in outlinks:
        if doc := cw22.get_doc(outlink):
            print(f"outlink doc {outlink} found")
            print(doc)
        else:
            print(f"outlink doc {outlink} not found")


if __name__ == "__main__":
    main()
