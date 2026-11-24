"""Return one private tool-assisted observation to cognition."""
from pm_workflows.python_role import RoleContext
from workflows.tool_call.steps import ToolRoleResult, execute_goal


def run(context: RoleContext) -> ToolRoleResult:
    return execute_goal(context, private=True)
