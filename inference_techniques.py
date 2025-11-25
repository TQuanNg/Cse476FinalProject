from utils import call_model_chat_completions

class InferenceTechnique:
    def __init__(self, inference_technique):
        self.call_counter = 0
        self.max_calls = 20
        self.inference_technique = inference_technique

    def _call(self, prompt: str, temperature: float = 0.0, system: str | None = None) -> str:
        if self.call_counter >= self.max_calls:
            return "ERROR: max call limit reached"
        self.call_counter += 1
        resp = call_model_chat_completions(
            prompt,
            system=system or "You are a helpful assistant. Reply with only the final answer—no explanation.",
            temperature=temperature,
        )
        if not resp.get("ok"):
            return f"ERROR status={resp.get('status')} {resp.get('error')}"
        return (resp.get("text") or "").strip()

    def chain_of_thought(self, question: str) -> str:
        cot = self._call(
            f"Think step-by-step and then provide the final concise answer.\nQUESTION: {question}\nRespond as: Final Answer: <answer>",
            temperature=0.7,
        )
        if "Final Answer:" in cot:
            return cot.split("Final Answer:")[-1].strip()
        return cot.strip()

    # First technique: Self-Refinement
    def self_refinement(self, question):
        answer = self._call(
            f"Answer the question clearly and concisely.\n\nQUESTION: {question}"
        )

        for _ in range(2):
            critique = self._call(
                f"You are a strict reviewer. Analyze the answer below and list any mistakes, "
                f"missing reasoning, incorrect logic, or unclear explanation.\n\n"
                f"QUESTION: {question}\n"
                f"ANSWER: {answer}\n"
                f"Respond with a short critique."
            )

            refined = self._call(
                f"Improve the answer using the critique below. Fix any errors, clarify reasoning, "
                f"and produce the best possible final answer.\n\n"
                f"QUESTION: {question}\n"
                f"CRITIQUE: {critique}\n"
                f"Give ONLY the improved final answer."
            )


            # If the refinement produced something new, update answer
            if refined and refined.strip() != answer.strip():
                answer = refined

        return answer


    # Second technique: Self consistency
    #
    def self_consistency(self, question, samples=3):
        chain_of_thought_answers = []

        # Generate many chains of thought with answers
        for _ in range(4):
            chain_of_thought = self._call(
                f"Solve step-by-step, but end with: 'Final Answer: <answer>'.\n\n"
                f"QUESTION: {question}",
                temperature=1.0
            )

            final = None
            if "Final Answer:" in chain_of_thought:
                final = chain_of_thought.split("Final Answer:")[-1].strip()
            else:
                final = chain_of_thought.strip() # if failed

            chain_of_thought_answers.append(final) # get all answers

        final_answer = max(set(chain_of_thought_answers), key=chain_of_thought_answers.count)

        return final_answer

    # Third technique: ReAct
    #
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
            f"Choose an ACTION.\n"
            f"Examples: SEARCH, LOOKUP, CALCULATE, CHECK_FACT, NO_ACTION.\n"
            f"Reply in a JSON-like format: ACTION: <action>"
        )
        observation = self._call(
            f"You decided on ACTION: {action}.\n"
            f"Simulate the OBSERVATION (result of the action).\n"
            f"Do NOT give final answer yet.\n"
            f"Reply starting with OBSERVATION: ..."
        )

        final = self._call(
            f"QUESTION: {question}\n"
            f"THOUGHT: {thought}\n"
            f"ACTION: {action}\n"
            f"OBSERVATION: {observation}\n"
            f"Now give ONLY the final answer.\n"
            f"Do not include chain-of-thought or steps."
        )

        return final

    # Alternative technique: Chain of Thought