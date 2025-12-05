from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    def solve_and_answer(self, question):

        qtype = self.technique.classify_question(question)
        print(f"++++Detected question type: {qtype} ++++")

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
            print(f"Unknown question type '{qtype}', using default chain of thought")
            return self.technique.chain_of_thought(question)

    def is_expression_task(self, question: str) -> bool:
        q = question.lower()
        return any(k in q for k in [
            "24-game",
            "use each number",
            "form an expression",
            "output format",
            "using + - * /",
        ])

    # math: use self-refinement as primary, with optional self-consistency verification
    def solve_math_question(self, question):

        print("\n===[Domain Handler] Using CoT and continuation prompting for math question===\n")

        if self.is_expression_task(question):
            return self.technique.solve_expression_question(question)
        else:
            return self.technique.solve_math_question(question)


    # common_sense: use chain of thought as primary
    def solve_commonsense_question(self, question):

        print("\n===[Domain Handler] Using ReAct for commonsense question===\n")

        cot_answer = self.technique.react(question)

        return cot_answer

    # future_prediction: use self-consistency as primary
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

    # coding: use self-refinement as primary
    def solve_coding_question(self, question):

        print("[Domain Handler] Using Self-Refinement for CODING question")

        refined_answer = self.technique.self_refinement_coding(question)

        return refined_answer
