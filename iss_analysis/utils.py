import inspect
import subprocess


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


def get_git_description(repo_path):
    """
    Get the git description (tag, commit, dirty status) of a repository.
    Falls back to commit hash if describe fails.
    """
    try:
        # Try to get a descriptive version (includes tags and dirty status)
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        # Fallback to just the commit hash if describe fails (e.g., no tags)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
