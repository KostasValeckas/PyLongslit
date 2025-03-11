Installation
~~~~~~~~~~~~~~

test

.. note::
    The steps in this documentation have been tested on 
    Linux Ubuntu 22.04.4 LTS (which should also cover MacOS functionality
    in broad sense) and Windows 10 Pro.

The software is run by executing a series of 
Python scripts. Therefore, the installation consists of these steps:


.. _Download the source code:

1. :ref:`Download the source code <download>`.
2. :ref:`Create a clean Python environment <venv>` (**optional - but strongly recommended**).
3. :ref:`Install the required Python packages <packages>`.

.. _download:

We provide both a quickgide for experienced python users and a more detailed guide for users who are new to Python environments.

Quick guide for experienced python users
----------------------------------------

1. `Download the source code. <https://github.com/KostasValeckas/PyLongslit/>`_
2. Create a new Python 3.10 environment.
3. Install the requirments from `requirments.txt` using ``pip``. 

Detailed guide for users new to Python environments
----------------------------------------------------

1. Download the source code
===========================

The software can be downloaded from the `GitHub repository website <https://github.com/KostasValeckas/PyLongslit/>`_ or 
cloned by using git.

**Direct link to download from the repository website:**

`Download Source Code (ZIP) <https://github.com/KostasValeckas/PyLongslit/archive/refs/heads/main.zip>`_

**Using git: (if you don't know what git is just download the ZIP from link above)** 

SSH (recommended if you plan on developing)...

.. code-block:: bash

    git clone git@github.com:KostasValeckas/PyLongslit.git

... or HTTPS (works too, but you will need to enter your username and password on every pull/push):

.. code-block:: bash

    git clone https://github.com/KostasValeckas/PyLongslit.git


.. _venv:

2. Create a clean Python environment
====================================

To ensure best possible stability of the software and to avoid version conflicts with other Python packages on your system,  
it is **strongly recommended** to create a clean Python environment for running the software.
If you are unfamiliar with Python environments, see :ref:`our quick introduction to
Python environments <envs_quick_into>`. You can skip directly to :ref:`installing the required Python packages <packages>`
if you prefer to not use a clean environment - in that case you might experience
software bugs due to version conflicts that are not accounted for in this documentation.

**Using Anaconda (conda) (recommeded):**

To create a new virtual environment using Anaconda, run the following command in your terminal 
(if you are using Windows, do this and all following from the Anaconda Prompt):

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

You can replace ``PyLongslit`` with any name you like. This will create a new environment with the Python version 3.10 installed.

.. note::

    If you are using Windows, you might need to run the following command instead:

    .. code-block:: powershell

        python -m venv PyLongslit

    This is because the Python executable might not be named ``python3.10`` on Windows.
    In that case, you can ensure that the correct version of Python is used by running:

    .. code-block:: powershell

        python --version

    If the Python version printed is not 3.10, you have several options:

    1. If your version is > 3.10, you most likely will be fine. Otherwise, try one of the following steps.
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
Feel free to* :ref:`skip to next section <packages>`.

Python applications often depend on a specific version of Python and a specific set of Python packages.
These packages can have dependencies on other packages, and these dependencies can have dependencies on other packages, and so on.
This can lead to a situation where two applications require different versions of the same package, which can cause conflicts.
By using Python environments, you can create isolated environments where you can install the specific versions of Python and Python 
packages that you need for a specific application. This helps ensure that only the needed packages are installed, and that they do not
conflict with other applications on your system. Furtermore, this ensures that 
updates to any packages do not break the application, as the environment will not be updated unless you explicitly update it.

Python environments as created with the commands shown above will be empty, and you will need to install the required packages,
as described in :ref:`installing the required Python packages <packages>`.

**Note:** The envirornment will need to be activated every time you open a new terminal.
You can configure your terminal to automatically activate the environment - 
this will not be covered in this documentation - see documentation for your terminal for more information.

**Example:**

In `bash` (Linux/MacOS), using `conda`, prior to activating a specific environment, 
your terminal will start in the `base` environment:

.. code-block:: bash

    (base) user@computer:~$

After activating the environment, the name of the environment will be shown in the terminal prompt:

.. code-block:: bash

    (PyLongslit) user@computer:~$

.. _packages:

3. Install the required Python packages
=======================================

To install the required Python packages, open your terminal and navigate to the directory 
where you downloaded the source code. Then run the following command:

(If you are using a clean Python environment, make sure you activate it first.)

.. code-block:: bash

    pip install -r requirements.txt

