"""Expose the main-cognition package tests to the root unittest discovery.

The tests live beside the workflow (``workflows/main_cognition``). Root
discovery runs ``discover -s tests``, so this module imports those classes into
a discovered module under unique names. Class names repeat across stages, so
each import is aliased.
"""
from workflows.main_cognition.tests.test_pipeline_contract import (  # noqa: F401
    DispatchTests as MainCognitionDispatchTests,
    KernelTests as MainCognitionPipelineKernelTests,
    PipelineTests,
)
from workflows.main_cognition.stages.triage.tests.test_triage_contract import (  # noqa: F401
    KernelTests as TriageKernelTests,
    MessageTests as TriageMessageTests,
)
from workflows.main_cognition.stages.appraisal.tests.test_appraisal_contract import (  # noqa: F401
    KernelTests as AppraisalKernelTests,
    MessageTests as AppraisalMessageTests,
)
from workflows.main_cognition.stages.context.tests.test_context_contract import (  # noqa: F401
    KernelTests as ContextKernelTests,
    MessageTests as ContextMessageTests,
    SelectionTests as ContextSelectionGuardTests,
)
from workflows.main_cognition.stages.attention.tests.test_attention_contract import (  # noqa: F401
    KernelTests as AttentionKernelTests,
    MessageTests as AttentionMessageTests,
    ValidationGuardTests as AttentionValidationGuardTests,
)
from workflows.main_cognition.stages.goals.tests.test_goals_contract import (  # noqa: F401
    KernelTests as GoalsKernelTests,
    MessageTests as GoalsMessageTests,
    ValidationGuardTests as GoalsValidationGuardTests,
)
from workflows.main_cognition.stages.workspace.tests.test_workspace_contract import (  # noqa: F401
    KernelTests as WorkspaceKernelTests,
    MessageTests as WorkspaceMessageTests,
    ValidationGuardTests as WorkspaceValidationGuardTests,
)
from workflows.main_cognition.stages.action_selection.tests.test_action_selection_contract import (  # noqa: F401,E501
    KernelTests as ActionSelectionKernelTests,
    MessageTests as ActionSelectionMessageTests,
    ValidationGuardTests as ActionSelectionValidationGuardTests,
)

__all__ = [
    "ActionSelectionKernelTests",
    "ActionSelectionMessageTests",
    "ActionSelectionValidationGuardTests",
    "AppraisalKernelTests",
    "AppraisalMessageTests",
    "AttentionKernelTests",
    "AttentionMessageTests",
    "AttentionValidationGuardTests",
    "ContextKernelTests",
    "ContextMessageTests",
    "ContextSelectionGuardTests",
    "GoalsKernelTests",
    "GoalsMessageTests",
    "GoalsValidationGuardTests",
    "MainCognitionDispatchTests",
    "MainCognitionPipelineKernelTests",
    "PipelineTests",
    "TriageKernelTests",
    "TriageMessageTests",
    "WorkspaceKernelTests",
    "WorkspaceMessageTests",
    "WorkspaceValidationGuardTests",
]
