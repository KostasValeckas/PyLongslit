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
            'pylongslit_check_config = pylongslit.check_config:main',
            'pylongslit_bias = pylongslit.mkspecbias:main',  
            'pylongslit_identify_arcs = pylongslit.identify:main',
            'pylongslit_combine_arcs = pylongslit.combine_arcs:main',
            'pylongslit_flat = pylongslit.mkspecflat:main',
            'pylongslit_crr = pylongslit.crremoval:main',
            'pylongslit_reduce = pylongslit.reduce:main',
            'pylongslit_crop = pylongslit.crop:main',
            'pylongslit_extract_1d = pylongslit.extract_1d:main',
            'pylongslit_extract_simple_1d = pylongslit.extract_simple_1d:main',
            'pylongslit_flux = pylongslit.flux_calibrate:main',
            'pylongslit_combine_spec = pylongslit.combine:main'
        ],
    },
)