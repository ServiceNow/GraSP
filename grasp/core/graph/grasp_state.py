from typing import TypedDict


class GraspState(TypedDict, total=False):
    """
    This class defines predefined state variables/schema.

    This will be implemented in derived class, which will be used to inject other state variables.
    Base class is only added for readability and keeping langgraph decoupled from platform
    """

    pass
