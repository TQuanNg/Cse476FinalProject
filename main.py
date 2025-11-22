import json
from utils import call_model_chat_completions, evaluate_tests, self_evaluate_tests, tests, MODEL
from agent import WorkingAgent

def load_data(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def run_agent():
    agent = WorkingAgent()

    question = "Explain why the sky is blue."
    answer = agent.solve_and_answer(question)

    print("\n=== Agent Final Answer ===")
    print(answer)
    return answer

def run_api_tests():
    print("\n=== Running direct API test ===")
    demo_prompt = "What is 17 + 28? Answer with just the number."
    result = call_model_chat_completions(demo_prompt)
    print("OK:", result["ok"], "HTTP:", result["status"])
    print("MODEL SAYS:", (result["text"] or "").strip())

    print("\n=== Running evaluation tests ===")
    evaluate_tests(tests)

    print("\n=== Running LLM-judge tests ===")
    self_evaluate_tests(tests, verbose=True, model=MODEL)

def main():
    print("=== Creating Agent ===")
    run_agent()

    print("\n=== Running Baseline API Tests ===")
    run_api_tests()

if __name__ == "__main__":
    main()