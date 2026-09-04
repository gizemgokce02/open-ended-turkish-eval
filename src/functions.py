from pathlib import Path
from typing import Literal
import json
import unicodedata
from vnlp import Normalizer, StopwordRemover, StemmerAnalyzer
import re


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

def normalize_text(text: str) -> str:
    """
    Normalize text for Turkish language processing.
    Parameters:
        text (str): The input text to normalize.
    Returns:
        str: The normalized text.
    """
    text = text.lower()
    text = unicodedata.normalize("NFC", text)

    # Fix Turkish ASCII-related character issues
    normalizer = Normalizer()
    text = normalizer.deasciify(text.split())
    text = " ".join(text)

    # Whitespace
    text = normalizer.remove_punctuations(text)
    text = normalizer.remove_accent_marks(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    stemming: bool = False
) -> str:
    """
    Preprocess text for Turkish language processing.
    Parameters:
        text (str): The input text to preprocess.
        remove_stopwords (bool): Whether to remove stopwords. Default is False.
        stemming (bool): Whether to apply stemming. Default is False.
    Returns:
        str: The preprocessed text.
    """
    text = normalize_text(text)

    if remove_stopwords:
        stopword_remover = StopwordRemover()
        tokens = text.split()
        tokens = stopword_remover.drop_stop_words(tokens)
        text = " ".join(tokens)

    if stemming: 
        stemmer = StemmerAnalyzer()
        text = " ".join(stemmer.predict(text))

    return text


def jaccard_similarity(answer: str, reference: str) -> float:
    """
    Calculate Jaccard similarity between two strings.
    Parameters:
        answer (str): The answer string.
        reference (str): The reference string.
    Returns:
        float: Jaccard similarity score (0-100).
    """
    answer_set = set(answer.split())
    reference_set = set(reference.split())

    union = answer_set | reference_set

    if not union:
        return 0.0

    return len(answer_set & reference_set) / len(union) * 100 # 0-100

