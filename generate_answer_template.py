#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Replace the placeholder logic inside `build_answers()` with your own agent loop
before submitting so the ``output`` fields contain your real predictions.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from agent import WorkingAgent


INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data

def is_placeholder(answer_text: str) -> bool:
    return answer_text.startswith("Placeholder answer")

def load_answers(path: Path, total: int) -> List[Dict[str, str]]:
    if not path.exists():
        return [{"output": f"Placeholder answer for question {i+1}"} for i in range(total)]

    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if len(data) != total:
        raise ValueError("answers.json length does not match questions.json length")

    return data


def build_answers(questions: List[Dict[str, Any]],
                  start_idx: int,
                end_idx: int) -> List[Dict[str, str]]:
    answers = []
    agent = WorkingAgent()

    SAVE_EVERY = 30

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            answers = json.load(f)
        print(f"Loaded {len(answers)} previous answers.")
    else:
        answers = []

    for idx in range(start_idx, end_idx + 1):
        current = answers[idx - 1]["output"]

        if not is_placeholder(current):
            print(f"Skipping Q{idx} (already solved)")
            continue

        try:
            question_input = questions[idx - 1].get("input", "")
            if not question_input:
                question_input = str(questions[idx - 1])

            agent.technique.call_counter = 0

            real_answer = agent.solve_and_answer(question_input)
            answers[idx - 1] = {"output": real_answer}
            print(f"Processed question {idx}/{len(questions)}")

        except Exception as e:
            # Fallback to placeholder if agent fails
            print(f"Error processing question {idx}: {e}")
            placeholder_answer = f"Error processing question {idx}: {str(e)}"
            answers[idx - 1] = {"output": placeholder_answer}

        if idx % SAVE_EVERY == 0:
            with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                json.dump(answers, f, ensure_ascii=False, indent=2)
            print("Checkpoint saved.")

    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )



START_INDEX = 1
END_INDEX = 6208

def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions, START_INDEX, END_INDEX)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r", encoding="utf-8") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()

