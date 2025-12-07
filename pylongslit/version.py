def get_version():
    """Get the version from setuptools metadata or fallback."""
    try:
        from importlib.metadata import version
        return version("pylongslit")
    except ImportError:
        # Fallback for Python < 3.8
        try:
            import pkg_resources
            return pkg_resources.get_distribution("pylongslit").version
        except:
            return "unknown"