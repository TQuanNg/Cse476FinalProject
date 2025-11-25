from inference_techniques import InferenceTechnique

class WorkingAgent:
    def __init__(self):
        self.technique = InferenceTechnique(self)

    def solve_and_answer(self, question):

        reasoning = self.technique.self_refinement(question)

        consistent_answer = self.technique.self_consistency(
            f"Based on this reasoning, give the final answer: {reasoning}",
            samples=3
        )
        refined_answer = self.technique.self_refinement(
            f"Question: {question}\nProposed answer: {consistent_answer}"
        )

        return refined_answer



