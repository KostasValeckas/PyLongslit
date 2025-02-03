from setuptools import setup, find_packages

setup(
    name='pylongslit',
    version='0.1',
    packages=find_packages(),  # Automatically find and include all your modules
    install_requires=[          # External dependencies
        'numpy',                # Example dependency
        'scipy',                # Example dependency
    ],
    entry_points={             # Define your CLI tools here
        'console_scripts': [
            'pylongslit_bias = pylongslit.mkspecbias:run_bias',  # This points to the function you want to run in CLI
        ],
    },
)