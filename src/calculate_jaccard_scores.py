import json
from pathlib import Path

from functions import jaccard_similarity, preprocess_text


DATA_PATH = Path("datasets/data01.json")


def calculate_jaccard_score(
    question_id: int,
    candidate_text: str,
    data_path: Path = DATA_PATH,
) -> float:
    """
    Calculate the Jaccard similarity between a candidate answer and
    the reference answer(s) of a question, then save the result
    directly into the original dataset.

    If multiple reference answers exist, the highest score is used.

    Args:
        question_id: ID of the question.
        candidate_text: Candidate/model answer.
        data_path: Path to the dataset.

    Returns:
        Best Jaccard similarity score in the range [0, 100].
    """

    # Load dataset
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load dataset: {e}") from e

    # Find question
    qa = None

    for dataset in data.get("data", []):
        for paragraph in dataset.get("paragraphs", []):
            for item in paragraph.get("qas", []):
                if item.get("id") == question_id:
                    qa = item
                    break

            if qa is not None:
                break

        if qa is not None:
            break

    if qa is None:
        raise ValueError(f"Question with ID {question_id} not found.")

    # Get reference answers
    references = [
        answer.get("text", "")
        for answer in qa.get("answers", [])
    ]

    if not references:
        raise ValueError(
            f"Question with ID {question_id} has no reference answers."
        )

    # Preprocess candidate
    processed_candidate = preprocess_text(candidate_text)

    # Calculate score against every reference
    scores = []

    for reference in references:
        processed_reference = preprocess_text(reference)

        score = jaccard_similarity(
            processed_candidate,
            processed_reference,
        )

        scores.append(score)

    # Use the best matching reference
    best_score = max(scores)

    # Create outputs list if it doesn't exist
    if "outputs" not in qa:
        qa["outputs"] = []

    # Append candidate + score
    qa["outputs"].append(
        {
            "text": candidate_text,
            "jaccard_score": float(best_score),
        }
    )

    # Overwrite original dataset
    try:
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4,
            )
    except OSError as e:
        raise RuntimeError(f"Failed to save dataset: {e}") from e

    return float(best_score)


if __name__ == "__main__":
    question_id = int(input("Question ID: "))
    candidate_text = input("Candidate answer: ")

    score = calculate_jaccard_score(
        question_id=question_id,
        candidate_text=candidate_text,
    )

    print(f"Jaccard similarity: {score:.2f}")