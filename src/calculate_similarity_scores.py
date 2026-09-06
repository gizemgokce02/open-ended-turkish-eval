import json
from pathlib import Path

from functions import cosine_similarity_score, preprocess_text


DATA_PATH = Path("datasets/data01.json")


def calculate_similarity_score(
    question_id: int,
    candidate_text: str,
    data_path: Path = DATA_PATH,
) -> float:
    """
    Calculate TF-IDF cosine similarity between a candidate answer
    and the reference answer(s) of a question.

    If multiple reference answers exist, the highest cosine
    similarity score is used.

    The result is stored under:
        qa["outputs"]["cosine_scores"]

    Only the matching QA object is modified.

    Args:
        question_id: ID of the question.
        candidate_text: Candidate/model answer.
        data_path: Path to the dataset.

    Returns:
        Best cosine similarity score in the range [0, 100].
    """

    # Load dataset
    try:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load dataset: {e}") from e

    # Find the requested question
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
        raise ValueError(
            f"Question with ID {question_id} not found."
        )

    # Get reference answers
    references = [
        answer.get("text", "")
        for answer in qa.get("answers", [])
        if answer.get("text")
    ]

    if not references:
        raise ValueError(
            f"Question with ID {question_id} has no reference answers."
        )

    # Preprocess candidate once
    processed_candidate = preprocess_text(candidate_text)

    cosine_scores = []

    # Compare candidate against every reference
    for reference in references:
        processed_reference = preprocess_text(reference)

        score = cosine_similarity_score(
            processed_candidate,
            processed_reference,
        )

        cosine_scores.append(score)

    # Use the best matching reference
    best_score = max(cosine_scores)

    # Create outputs object if it doesn't exist
    if "outputs" not in qa:
        qa["outputs"] = {}

    # Create cosine_scores list if it doesn't exist
    if "cosine_scores" not in qa["outputs"]:
        qa["outputs"]["cosine_scores"] = []

    # Append candidate and score
    qa["outputs"]["cosine_scores"].append(
        {
            "text": candidate_text,
            "cosine_score": float(best_score),
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
        raise RuntimeError(
            f"Failed to save dataset: {e}"
        ) from e

    return float(best_score)


if __name__ == "__main__":
    question_id = int(input("Question ID: "))
    candidate_text = input("Candidate answer: ")

    score = calculate_similarity_score(
        question_id=question_id,
        candidate_text=candidate_text,
    )

    print(f"TF-IDF Cosine similarity: {score:.2f}")