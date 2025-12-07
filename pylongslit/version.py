from importlib.metadata import version, PackageNotFoundError

def get_version():
    """Get the version from setuptools metadata."""
    try:
        return version("pylongslit")
    except PackageNotFoundError:
        return "unknown"