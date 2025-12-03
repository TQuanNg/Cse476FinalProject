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
        
            DO NOT ANSWER THE QUESTION, ONLY CLASSIFY IT.
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

        #print("[Self-Consistency] Answers:", answers)

        # Majority vote
        final_answer = max(set(answers), key=answers.count)
        confidence = answers.count(final_answer) / len(answers)

        #print(f"[Self-Consistency] Final = {final_answer} (confidence={confidence:.2f})")

        return final_answer

    def future_consistency(self, question, samples=4):
        predictions = []

        for _ in range(samples):
            response = self._call(
                f"""
                    {question}

                    IMPORTANT:
                    Your final answer MUST end with this exact format:
                    \\boxed{{YOUR_PREDICTION}}
                    Do not add anything else.
                    """,
                temperature=0.8
            )

            answer = response.strip()
            if "\\boxed{" in answer:
                answer = answer[answer.find("\\boxed{"):]  # remove extra text
                answer = answer.splitlines()[0].strip()  # first line only

            predictions.append(answer)

        # print("[Future-Prediction Probabilities] Raw predictions:", predictions)

        # Count frequencies
        from collections import Counter
        count = Counter(predictions)
        most_common = count.most_common(1)[0][0]
        confidence = count[most_common] / len(predictions)

        # print(f"[Future-Prediction Probabilities] Selected = {most_common} (confidence={confidence:.2f})")

        return most_common

    # Used for commonsense
    def react(self, question):
        thought = self._call(
            f"You are an agent using the ReAct pattern.\n"
            f"THOUGHT: Think step-by-step about the question.\n"
            f"Do NOT answer yet.\n"
            f"QUESTION: {question}\n"
            f"Respond with only your chain-of-thought as THOUGHT: ..."
        )

        action = self._call(
            f"Based on the THOUGHT:\n{thought}\n\n"
            f"Proceed to perform an ACTION to help answer the question and retrieve all RELEVANT contexts TO that question.\n"
            f"Action should be done in many subjects in the question."
            f" Recommended amount of action is 2, maximum amount of action is 4\n"
            f"Some examples of ACTIONS you can take are: Search[query], Calculate[equation], Lookup[topic].\n"
        )

        #print(f"[React] thought: {thought}\n")
        #print(f"[React] action: {action}\n")

        observation = self._call(
            f"Based on context from {action}.\n"
            f"for answering the QUESTION: {question}\n"
            f"Perform those ACTIONS."
            f"Then perform any observations or calculations needed. Avoid false facts.\n"
            f"If direct action does not give enough info to determine the answer, then a logical deduction must be done.\n"
            f"Do NOT give final answer yet.\n"
        )

        final = self._call(
            f"QUESTION: {question}\n"
            f"THOUGHT: {thought}\n"
            f"ACTION: {action}\n"
            f"OBSERVATION: {observation}\n"
            f"Now give ONLY a brief final answer."
            f"No need for a full sentence answer."
            f"If it is a name, give full name."
            f"Do not include chain-of-thought or steps."
        )

        #print(f"[React] observation: {observation}\n")
        #print(f"[React] answer: {final}\n")

        return final

    # First technique for solving math problem: chain of thought
    # Output is step by step solution
    def chain_of_thought_math(self, question: str) -> str:
        prompt = f"""
            You are a professional mathematician. Be concise and strictly symbolic.
        
            OUTPUT FORMAT (MANDATORY):
            Step 1: <short equation or statement>
            Step 2: <short equation or statement>
            ...
            Final Answer: <number>
        
            RULES:
            - Define variables before first use.
            - Use only short lines (one equation or one small derivation per step).
            - Do NOT include paragraphs or storytelling.
            - Do NOT assume special shapes/angles/symmetry unless provable from data.
            - If a claim cannot be proven from the problem data, state: "Step X: CANNOT_PROVE: <brief reason>".
            - Use minimal language to save tokens.
        
            QUESTION:
            {question}
        
            Remember: Output must follow the exact format above.
        """
        # Lower temperature for deterministic math outputs
        cot = self._call(prompt, temperature=0.2)

        # If model failed to follow format, try to salvage by forcing minimal cleanup
        if "Step 1:" not in cot and "Final Answer:" in cot:
            # wrap the whole thing into a single step to allow patching,
            cot = "Step 1: " + cot.replace("\n", " ") + "\nFinal Answer: " + (
                re.search(r"Final Answer:\s*(.*)", cot, re.IGNORECASE).group(1).strip()
                if re.search(r"Final Answer:\s*(.*)", cot, re.IGNORECASE)
                else ""
            )
        return cot.strip()

    # Based on chain_of_thought_math, iteratively refine until solved
    def solve_math_question(self, question, max_iters: int = 2):
        full_solution = self.chain_of_thought_math(question)
        print(f"[Solver] Initial output:\n{full_solution}\n")


    # Refining CoT for better answer
    def self_refinement_coding(self, question):
        answer = self.chain_of_thought_coding(question)
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

    # CoT for coding problems
    def chain_of_thought_coding(self, question: str) -> str:
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
