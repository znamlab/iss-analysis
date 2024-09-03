import inspect


def get_default_args(func):
    """Get the default arguments of a function.

    Args:
        func (callable): The function to get the default arguments of.

    Returns:
        dict: A dictionary of the default arguments.
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
