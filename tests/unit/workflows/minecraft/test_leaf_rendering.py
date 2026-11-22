import unittest

from pm.context_render.prompt import RenderSettings
from workflows.minecraft.execution_contracts import LeafProject, LeafSpecification
from workflows.minecraft.rendering import LeafRenderInput, render_leaf_specification
from workflows.minecraft_project.planning.context_contracts import ProjectIntent
from workflows.minecraft_project.schemas import Condition, Node


class LeafRenderingTests(unittest.TestCase):
    def test_repeated_parent_goal_fits_alongside_distinct_leaf_requirements(self) -> None:
        goal = "\n".join(f"Inspect block {x}, 64, 7." for x in range(500))
        criterion = Condition(id="safe", description="Stay alive during inspection.", kind="alive")
        root = Node(id="survey", kind="goal", instruction=goal, criteria=[criterion])
        constraint = "Preserve every inspected block."
        leaf = LeafSpecification(identity=None, instruction="Inspect the first row.", criteria=(criterion,),
            project_intent=ProjectIntent(original_request=goal, original_goal="", current_instruction=goal,
                constraints=(constraint,)),
            current_criterion_observations=None, project=LeafProject(goal=goal, ancestor_specifications=(root,)))
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=8192)
        prompt = render_leaf_specification(LeafRenderInput(leaf), settings)
        self.assertEqual(prompt.raw_text.count(goal), 1)
        self.assertIn(f"Parent project goal:\n{goal}", prompt.raw_text)
        self.assertIn("Original request: Same as Parent project goal.", prompt.raw_text)
        self.assertIn("Current instruction: Same as Parent project goal.", prompt.raw_text)
        self.assertIn("\nSame as Parent project goal.\nRequired outcomes:", prompt.raw_text)
        self.assertIn(constraint, prompt.raw_text)
        self.assertIn(leaf.instruction, prompt.raw_text)
        self.assertEqual(prompt.raw_text.count(criterion.description), 2)

    def test_amended_instructions_remain_verbatim(self) -> None:
        goal = "Inspect the blocks along the north wall."
        amendment = goal + " Also check the door."
        criterion = Condition(id="safe", description="Stay alive.", kind="alive")
        root = Node(id="survey", kind="goal", instruction=amendment, criteria=[criterion])
        intent = ProjectIntent(original_request=goal, original_goal="", current_instruction=amendment, constraints=())
        leaf = LeafSpecification(identity=None, instruction="Inspect the hinges.", criteria=(criterion,),
            project_intent=intent, current_criterion_observations=None,
            project=LeafProject(goal=goal, ancestor_specifications=(root,)))
        settings = RenderSettings(context_window=32000, input_fraction=0.8, reserved_output_tokens=8192)
        prompt = render_leaf_specification(LeafRenderInput(leaf), settings)
        self.assertIn(f"Current instruction: {amendment}\n", prompt.raw_text)
        self.assertIn(f"\n{amendment}\nRequired outcomes:", prompt.raw_text)
