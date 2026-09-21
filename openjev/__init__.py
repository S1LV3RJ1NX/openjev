"""openjev: typed decision models you can train on your own data.

State in, typed probabilistic answers out. No text generation, so nothing to
parse. The API surface mirrors TypeSafe's Jev deliberately, so a comparison is
apples-to-apples and existing Jev code ports without change.
"""

from openjev.schema import (
    Answer,
    Choice,
    Example,
    Noul,
    Question,
    Score,
    Task,
    question_from_dict,
)


def __getattr__(name: str):
    # DecisionModel pulls in torch and transformers. Importing it lazily keeps
    # `from openjev import Task` usable in the dataset-building venv, which
    # deliberately has neither.
    if name == "DecisionModel":
        from openjev.infer import DecisionModel

        return DecisionModel
    raise AttributeError(f"module 'openjev' has no attribute {name!r}")


__all__ = [
    "Answer",
    "Choice",
    "DecisionModel",
    "Example",
    "Noul",
    "Question",
    "Score",
    "Task",
    "question_from_dict",
]
