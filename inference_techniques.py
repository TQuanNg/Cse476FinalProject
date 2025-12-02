from typing import Optional, Tuple

from utils import call_model_chat_completions
import re

class InferenceTechnique:
    def __init__(self, inference_technique):
        self.call_counter = 0
        self.max_calls = 20
        self.inference_technique = inference_technique

    def _call(self, prompt: str, temperature: float = 0.0, token: int = 800, system: str | None = None) -> str:
        if self.call_counter >= self.max_calls:
            return "ERROR: max call limit reached"
        self.call_counter += 1
        resp = call_model_chat_completions(
            prompt,
            system=system or "You are a helpful assistant.",
            temperature=temperature,
        )
        if not resp.get("ok"):
            return f"ERROR status={resp.get('status')} {resp.get('error')}"
        return (resp.get("text") or "").strip()

    def classify_question(self, question):
        """
        Returns: 'math', 'commonsense', 'future_prediction', or 'planning'
        """
        prompt = f"""
            Classify the following question into ONE category:
            - math (requires calculation, equations, numbers, mathematical reasoning)
            - commonsense (real-world knowledge, everyday reasoning, general facts)
            - future_prediction (asking about what will happen, forecasting, predictions)
            - coding (requires programming, code generation, debugging)
            - planning (requires step-by-step planning, strategy, multi-step processes)
        
            QUESTION:
            {question}
        
            Return ONLY one word, NO EXTRA CHARACTER: math, commonsense, future_prediction, coding, or planning.
            """

        result = self._call(
            prompt,
            system="Return only one label: math, commonsense, future_prediction, coding, or planning.",
            temperature=0.0,
        )

        return (result or "").strip().lower()

    # Second technique: Self consistency
    #
    def self_consistency(self, question, samples=4):
        answers = []

        for i in range(samples):
            response = self._call(
                f"""
                    {question}

                    Solve carefully.
                    End your response with:
                    Final Answer: <your answer>
                    """,
                temperature=0.8
            )

            # Extract answer
            if "Final Answer:" in response:
                ans = response.split("Final Answer:")[-1].strip()
                ans = ans.split("\n")[0].strip()
            else:
                # fallback if model forgets format
                extract = self._call(
                    f"Extract ONLY the final answer from this:\n\n{response}",
                    system="Return only the answer.",
                    temperature=0.0
                )
                ans = extract.strip()

            answers.append(ans)

        print("[Self-Consistency] Answers:", answers)

        # Majority vote
        final_answer = max(set(answers), key=answers.count)
        confidence = answers.count(final_answer) / len(answers)

        print(f"[Self-Consistency] Final = {final_answer} (confidence={confidence:.2f})")

        return final_answer

    def self_refinement_coding(self, question):

        answer = self.chain_of_thought(question)
        print(f"[Self-Refinement] Initial Answer:\n{answer}\n")

        for i in range(2):
            verifier_prompt = f"""
        You are a strict code reviewer.

        TASK:
        Check if the following code fully satisfies the specification.

        QUESTION:
        {question}

        CODE:
        {answer}

        OUTPUT FORMAT (STRICT — ONE LINE ONLY):

        - If correct, output EXACTLY:
        VALID

        - If incorrect, output EXACTLY ONE LINE in this form:
        FIX: <short actionable correction instruction>

        Example:
        FIX: You forgot to return plt.gca()
        FIX: Salary must be random.randint(*SALARY_RANGE)

        NO explanation. NO bullets. ONE line only.
        """
            critique = self._call(verifier_prompt, temperature=0.0)
            print(f"[Self-Refinement] Iteration {i + 1} - Critique:\n{critique}\n")

            # Stop if correct
            if critique.strip().upper() == "VALID":
                print(f"[Self-Refinement] Code VALID at iteration {i + 1}.\n")
                break

            # ---- PATCHER ----
            patch_prompt = f"""
        You are in CORRECTION MODE.

        QUESTION:
        {question}

        CURRENT CODE:
        {answer}

        CRITIQUE:
        {critique}

        INSTRUCTIONS:
        - Apply ONLY the fix described in the critique.
        - Do NOT rewrite the entire solution.
        - Do NOT change working logic.
        - Preserve the exact function signature.
        - Output ONLY corrected Python code.
        """
            refined = self._call(patch_prompt, temperature=0.0)
            print(f"[Self-Refinement] Iteration {i + 1} - Refined Code:\n{refined}\n")

            # ---- SAFETY UPDATE ----
            if refined.strip():
                answer = refined.strip()
            else:
                print("[Self-Refinement] Empty patch received — aborting.\n")
                break

        print(f"[Self-Refinement] Final Code Used:\n{answer}\n")
        return answer

    # alternative method
    def chain_of_thought(self, question: str) -> str:
        """
        Initial code generator for coding tasks only.
        """
        prompt = f"""
    You are a professional Python developer.

    TASK:
    Generate a correct and minimal code solution for the following problem.

    OUTPUT RULES (MANDATORY):
    - Output ONLY Python code.
    - Include all required imports and constants.
    - Define the function exactly as requested.
    - Do NOT explain.
    - Do NOT include markdown.
    - Ensure the function returns the correct object.
    - Follow instructions literally (title, labels, return type, etc).

    QUESTION:
    {question}
    """
        code = self._call(prompt, temperature=0.25)
        return code.strip()
