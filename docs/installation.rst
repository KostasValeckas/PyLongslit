Installation
~~~~~~~~~~~~~~


.. note::
    The steps in this documentation have been tested on 
    Linux Ubuntu 24.04.2 LTS (which should also cover MacOS functionality
    in a broad sense) and Windows 10 Pro.

The software is run by executing a series of commands in a terminal.
When the software is installed with the steps below, the command
line tools will be available in the terminal.

The installation process consists of the following steps:

1. :ref:`Create a clean Python environment <venv>` (**optional - but strongly recommended**).
2. :ref:`Install the software using pip <install>`.


We provide both a quick guide for experienced Python users and a more detailed guide for users who are new to Python environments.

Lastly, we also describe how to download and :ref:`install an editable
version of the software <dev_install>` if you wish to contribute to the development of the software.

Quick guide for experienced Python users
-----------------------------------------

1. Create a new Python 3.10 environment.
2. Install using ``pip``:

.. code-block:: bash

    pip install pylongslit


Detailed guide for users new to Python environments
----------------------------------------------------

.. _venv:

1. Create a clean Python environment
====================================

To ensure the best possible stability of the software and to avoid version conflicts with other Python packages on your system,  
it is **strongly recommended** to create a clean Python environment for running the software.
If you are unfamiliar with Python environments, see :ref:`our quick introduction to
Python environments <envs_quick_into>`. You can skip directly to :ref:`installing the software <install>` if you prefer not to use a clean environment - in that case you might experience
software bugs due to version conflicts that are not accounted for in this documentation.

**Using Anaconda (conda) (recommended):**

To create a new virtual environment using Anaconda, run the following command in your terminal 
(if you are using Windows, do this and all following commands from the Anaconda Prompt):

.. code-block:: bash

    conda create --name PyLongslit python=3.10

You can replace ``PyLongslit`` with any name you like. This will create a new environment with Python 3.10 installed.

To activate the environment, run:

.. code-block:: bash

    conda activate PyLongslit

**Using venv (standard Python):**

To create a new virtual environment using venv (standard Python), make sure you have Python 3.10 installed,
then run the following command in your terminal:

.. code-block:: bash

    python3.10 -m venv PyLongslit

You can replace ``PyLongslit`` with any name you like. This will create a new environment with Python version 3.10 installed.

.. note::

    If you are using Windows, you might need to run the following command instead:

    .. code-block:: powershell

        python -m venv PyLongslit

    This is because the Python executable might not be named ``python3.10`` on Windows.
    In that case, you can ensure that the correct version of Python is used by running:

    .. code-block:: powershell

        python --version

    If the Python version printed is not 3.10, you have several options:

    1. If your version is not 3.10, you most likely will be fine. Otherwise, try one of the following steps.
    2. Install Anaconda and create the environment using the conda command as described above.
    3. You can set the Python version to be used by the terminal by adding the Python installation directory to the PATH environment variable. See the following link for more information: `How to set the path and environment variables in Windows <https://realpython.com/add-python-to-path/>`_.

To activate the environment, run:

For Linux/MacOS:

.. code-block:: bash

    source PyLongslit/bin/activate

, where ``PyLongslit/bin/activate`` is the path to the activate script in the environment.

For Windows:

.. code-block:: powershell

    # In PowerShell
    .\PyLongslit\Scripts\Activate.ps1


    # In cmd.exe
    .\PyLongslit\Scripts\Activate.bat

, where ``PyLongslit/Scripts`` is the path to the activate script in the environment.

.. _envs_quick_into:


Quick introduction to Python environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*The following is a quick introduction to Python environments for users who would like one.
Feel free to* :ref:`skip to the next section <install>`.

Python applications often depend on a specific version of Python and a specific set of Python packages.
These packages can have dependencies on other packages, and these dependencies can have dependencies on other packages, and so on.
This can lead to a situation where two applications require different versions of the same package, which can cause conflicts.
By using Python environments, you can create isolated environments where you can install the specific versions of Python and Python 
packages that you need for a specific application. This helps ensure that only the needed packages are installed, and that they do not
conflict with other applications on your system. Furthermore, this ensures that 
updates to any packages do not break the application, as the environment will not be updated unless you explicitly update it.

**Note:** The environment will need to be activated every time you open a new terminal.
You can configure your terminal to automatically activate the environment upon startup. This will not be covered in this documentation - see the documentation for your terminal for more information.

**Example:**

In `bash` (Linux/MacOS), using `conda`, prior to activating a specific environment, 
your terminal will start in the `base` environment:

.. code-block:: bash

    (base) user@computer:~$

After activating the environment, the name of the environment will be shown in the terminal prompt:

.. code-block:: bash

    (PyLongslit) user@computer:~$

.. _install:

2. Install the software using pip
=======================================

To install the software and the required dependencies, 
run the following command in your terminal:

(if you are using a clean Python environment, make sure you activate it first.)

.. code-block:: bash

    pip install pylongslit

After the installation is complete, you can perform a quick check
to see if the software was installed by running the following command:

.. code-block:: bash

    pylongslit_check_config --help

If the software was installed correctly, you should see a message 
like this in the terminal:

.. code-block:: bash

    usage: pylongslit_check_config [-h] config

    Run the pylongslit config-file checker.

    positional arguments:
      config      Configuration file path

    options:
      -h, --help  show this help message and exit

.. _dev_install:

Developer installation
===========================

An editable version of the software can be installed if you plan on developing the PyLongslit code.
This allows you to make changes to the software and see the changes reflected in the command line tools without having to reinstall the software.

It is recommended that you `fork the repository <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ 
and develop from your own fork. If you think that your changes to the code would benefit all users of PyLongslit, feel free to create a `pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_.
We are always grateful for any contributions, but make sure your changes reflect :ref:`our software principles <index>`.

The below guide on installation shows how to clone the repository, and it will work for your own fork if you exchange 
the URL links to the links to your own fork. 


**Using git:** 

SSH (recommended - help on SSH keys can be found `here <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_):

.. code-block:: bash

    git clone git@github.com:KostasValeckas/PyLongslit.git

... or HTTPS (works too, but you will need to enter your username and password on every pull/push):

.. code-block:: bash

    git clone https://github.com/KostasValeckas/PyLongslit.git

**You can also** `download a snapshot of the repository as a ZIP file <https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github#downloading-a-repositorys-files>`_ 
, **but this is not recommended for developing.**

Then, when in :ref:`clean Python 3.10 environment <venv>`, 
navigate to the directory where the software was downloaded (this is the directory with the file `setup.cfg` in it) and run the following command:

.. code-block:: bash

    pip install -e .[all]


This will install the software in "editable" mode. The flag ``.[all]`` ensures that all optional dependencies are also installed. These will allow you to update/build the documentation and run the test suite.

There exists a `secondary code repository <https://github.com/KostasValeckas/PyLongslit_dev>`_ for holding test/tutorial data,
files needed to use the software for specific instruments, and other development-related files. It is recommended that you fork this 
repository as well if you plan on developing the software, and update it in tandem with your PyLongslit fork. 



