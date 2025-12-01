from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    # call to classify domain (math, common_sense, future_prediction, planning) of question:

    # math: use self-refinement as primary, with optional self-consistency verification
    def solve_math_question(self, question):
        print("\n===[Domain Handler] Using Self-Refinement for math question===\n")

        refined_answer = self.technique.solve_math_question(question)

        # Optional verification: Self-consistency with fewer samples for efficiency
        verification_result = self.technique.self_consistency(
            f"Verify this math solution:\n\nQuestion: {question}\n\nProposed Answer: {refined_answer}\n\nIs this correct? If not, provide the correct answer.",
            samples=3
        )

        if verification_result and verification_result.strip() != refined_answer.strip():
            print(f"[Math Handler] Verification suggests different answer: {verification_result}")
            return verification_result

        return refined_answer

    # common_sense: use chain of thought as primary
    def solve_commonsense_question(self, question):

        print("\n===[Domain Handler] Using Chain of Thought for commonsense question===\n")

        cot_answer = self.technique.react(question)

        return cot_answer

    def solve_future_prediction_question(self, question):
        print("\n===[Domain Handler] Using Self-Consistency for FUTURE PREDICTION question===\n")

        # Primary: Self-consistency with more samples for better prediction
        consistent_answer = self.technique.future_consistency(question, samples=4)

        return consistent_answer

    # planning: use ReAct as primary
    def solve_planning_question(self, question):
        print("\n===[Domain Handler] Using ReAct for PLANNING question===\n")

        # Primary: ReAct
        react_answer = self.technique.reasoning_via_planning(question)

        return react_answer

    def solve_coding_question(self, question):
        print("[Domain Handler] Using Self-Refinement for CODING question")

        refined_answer = self.technique.self_refinement(question)

        return refined_answer

    def solve_and_answer(self, question):
        qtype = self.technique.classify_question(question)
        print(f"[Router] Detected question type: {qtype}")

        # Route to appropriate domain-specific handler
        if qtype == "math":
            return self.solve_math_question(question)
        elif qtype == "commonsense":
            return self.solve_commonsense_question(question)
        elif qtype == "future_prediction":
            return self.solve_future_prediction_question(question)
        elif qtype == "planning":
            return self.solve_planning_question(question)
        elif qtype == "coding":
            return self.solve_coding_question(question)
        else:
            # Fallback for unknown question types
            print(f"[Router] Unknown question type '{qtype}', using default chain of thought")
            return self.technique.chain_of_thought(question)



