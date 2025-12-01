import os, json, textwrap, re, time
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL    = os.getenv("MODEL_NAME", "bens_model")

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 900,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

tests = [
    {
        "id": "math_inequality",
        "type": "numeric",  # grader will prefer numeric extraction
        "prompt": "Solve for the smallest integer n such that 3n + 5 > 26. Answer with just the integer.",
        "expected": "8",    # Because 3n > 21 => n > 7, smallest integer is 8
    },
    {
        "id": "commonsense_ice",
        "type": "text",
        "prompt": (
            "You place an ice cube in a glass of water and mark the water level. "
            "After the ice melts, does the water level rise, fall, or stay the same? "
            "Answer with exactly one of: 'rise', 'fall', 'stay the same'."
        ),
        "expected": "stay the same",
    },
    {
        "id": "logic_race",
        "type": "text",
        "prompt": (
            "In a race, you pass the person in second place. What position are you now in? "
            "Answer with a single word like 'first', 'second', 'third'."
        ),
        "expected": "second",
    },
]


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    # Remove surrounding punctuation and extra whitespace
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Map common synonyms used in these tests
    synonyms = {
        "unchanged": "stay the same",
        "no change": "stay the same",
        "same": "stay the same",
        "second place": "second",
        "2nd": "second",
        "first place": "first",
        "third place": "third",
    }
    return synonyms.get(s, s)

def extract_number(s: str):
    # Returns first number occurrence as string if found, else None
    if not s:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else None

def grade(expected: str, got: str, kind: str) -> bool:
    if kind == "numeric":
        exp_num = extract_number(expected)
        got_num = extract_number(got)
        return (exp_num is not None) and (got_num == exp_num)
    else:
        return normalize_text(got) == normalize_text(expected)

def evaluate_tests(tests, model=MODEL):
    rows = []
    for t in tests:
        r = call_model_chat_completions(
            t["prompt"],
            system="You are a careful solver. Reply ONLY with the final answer, nothing else.",
            model=model,
            temperature=0.0,
        )
        got = (r["text"] or "").strip()
        is_correct = grade(t["expected"], got, t["type"])
        rows.append({
            "id": t["id"],
            "expected": t["expected"],
            "got": got,
            "correct": is_correct,
            "status": r["status"],
            "error": r["error"],
        })
        # Tiny pacing to be polite to the API
        time.sleep(0.2)

    # Print a small report
    correct = sum(1 for x in rows if x["correct"])
    print(f"Score: {correct}/{len(rows)} correct")
    for x in rows:
        mark = "✅" if x["correct"] else "❌"
        print(f"{mark} {x['id']}: expected={x['expected']!r}, got={x['got']!r} (HTTP {x['status']})")
        if x["error"]:
            print("   error:", x["error"])
    return rows



def self_evaluate(question, prediction, expected_answer, model=MODEL):
    """
    Use the model itself as a strict grader.
    Returns True if the model says the prediction matches the expected answer; else False.
    Falls back to a simple normalized string compare if the model's reply is malformed.
    """
    import re

    system = "You are a strict grader. Reply with exactly True or False. No punctuation. No explanation."
    prompt = f"""You are grading a question-answer pair.

Return exactly True if the PREDICTION would be accepted as correct for the EXPECTED_ANSWER.
Otherwise, return False.

QUESTION:
{question}

PREDICTION:
{prediction}

EXPECTED_ANSWER:
{expected_answer}

Answer with exactly: True or False
"""

    r = call_model_chat_completions(
        prompt,
        system=system,
        model=model,
        temperature=0.0,
    )

    reply = (r.get("text") or "").strip().lower()
    if reply.startswith("true"):
        return True
    if reply.startswith("false"):
        return False

    # Fallback: simple normalization-based equality
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return norm(prediction) == norm(expected_answer)


def self_evaluate_tests(tests, model=MODEL, grader_model=None, sleep_sec=0.2, verbose=True):
    import time

    judge_model = grader_model or model
    rows = []

    for t in tests:
        # 1) Get model prediction
        r = call_model_chat_completions(
            t["prompt"],
            system="You are a careful solver. Reply ONLY with the final answer, nothing else.",
            model=model,
            temperature=0.0,
        )
        got = (r.get("text") or "").strip()

        # 2) LLM-as-a-judge: strict True/False
        is_correct = self_evaluate(
            question=t["prompt"],
            prediction=got,
            expected_answer=t["expected"],
            model=judge_model,
        )

        row = {
            "id": t.get("id", "<unnamed>"),
            "expected": t["expected"],
            "got": got,
            "correct": bool(is_correct),
            "status": r.get("status"),
            "error": r.get("error"),
        }
        rows.append(row)

        if verbose:
            mark = "✅" if is_correct else "❌"
            print(f"{mark} {row['id']}: expected={row['expected']!r}, got={row['got']!r} (HTTP {row['status']})")
            if row["error"]:
                print("   error:", row["error"])

        if sleep_sec:
            time.sleep(sleep_sec)

    return rows


def evaluate_tests_with_agent(tests, agent):
    import time

    rows = []
    for t in tests:
        # Reset the agent's call counter for each question
        agent.technique.call_counter = 0

        # Use agent to solve
        got = agent.solve_and_answer(t["prompt"])

        # Grade the answer
        is_correct = grade(t["expected"], got, t["type"])

        rows.append({
            "id": t["id"],
            "expected": t["expected"],
            "got": got,
            "correct": is_correct,
            "status": 200,
            "error": None,
        })

        # Tiny pacing to be polite to the API
        time.sleep(0.2)

    # Print a small report
    correct = sum(1 for x in rows if x["correct"])
    print(f"Score: {correct}/{len(rows)} correct")
    for x in rows:
        mark = "✅" if x["correct"] else "❌"
        print(f"{mark} {x['id']}: expected={x['expected']!r}, got={x['got']!r}")
        if x.get("error"):
            print("   error:", x["error"])
    return rows


def self_evaluate_tests_with_agent(tests, agent, grader_model=None, sleep_sec=0.2, verbose=True):
    import time

    judge_model = grader_model or MODEL
    rows = []

    for t in tests:
        # Reset the agent's call counter for each question
        agent.technique.call_counter = 0

        # 1) Get agent prediction
        got = agent.solve_and_answer(t["prompt"])

        # 2) LLM-as-a-judge: strict True/False
        is_correct = self_evaluate(
            question=t["prompt"],
            prediction=got,
            expected_answer=t["expected"],
            model=judge_model,
        )

        row = {
            "id": t.get("id", "<unnamed>"),
            "expected": t["expected"],
            "got": got,
            "correct": bool(is_correct),
            "status": 200,
            "error": None,
        }
        rows.append(row)

        if verbose:
            mark = "[OK ✅]" if is_correct else "[FAIL ❌]"
            print(f"{mark} {row['id']}: expected={row['expected']!r}, got={row['got']!r}")
            if row.get("error"):
                print("   error:", row["error"])

        if sleep_sec:
            time.sleep(sleep_sec)

    return rows


