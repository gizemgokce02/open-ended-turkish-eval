from pathlib import Path
from typing import Literal
import json
from vnlp import Normalizer, StopwordRemover, StemmerAnalyzer


normalizer = Normalizer()
stopword_remover = StopwordRemover()
stemmer = StemmerAnalyzer()

DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "data01.json"

DataType = Literal["all", "q", "a"]

_data: dict[int, dict] | None = None


def load_data(path: Path = DATA_PATH) -> dict[int, dict]:
    """Load and index Q&A data by question ID."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f).get("data", [])
    except (OSError, json.JSONDecodeError) as e:
        print(f"Failed to load data: {e}")
        return {}

    data = {}

    for dataset in raw_data:
        for paragraph in dataset.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                data[qa["id"]] = qa

    print(f"Loaded data: {len(data)} questions")
    return data


def get_data(id: int, type: DataType) -> list:
    """
    Get data by question ID and type.

    Types:
        all -> full Q&A object
        q   -> question text
        a   -> answer text(s)
    """
    global _data

    if _data is None:
        _data = load_data()

    item = _data.get(id)

    if item is None:
        return []

    if type == "all":
        return [item]

    if type == "q":
        return [item["question"]]

    if type == "a":
        return [answer["text"] for answer in item.get("answers", [])]

    raise ValueError("Type must be 'all', 'q', or 'a'")

id = 7112
print(get_data(id, "all"))
print(get_data(id, "q"))
print(get_data(id, "a"))

def preprocess_text(text: str) -> str:
    """
    Normalize, remove stopwords, and stem a Turkish text.
    """
    text = normalizer.lower_case(text)
    text = normalizer.deasciify(text.split())
    text = normalizer.remove_punctuations(" ".join(text))
    text = normalizer.remove_accent_marks(text)

    text = stopword_remover.drop_stop_words(text.split())
    text = stemmer.predict(" ".join(text))

    return " ".join(text)


def jaccard_similarity(answer: str, reference: str) -> float:
    """
    Calculate Jaccard similarity between two strings.
    """
    answer_set = set(answer.lower().split())
    reference_set = set(reference.lower().split())

    union = answer_set | reference_set

    if not union:
        return 0.0

    return len(answer_set & reference_set) / len(union)
