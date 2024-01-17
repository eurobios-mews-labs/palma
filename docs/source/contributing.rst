.. toctree::
   :maxdepth: 2
   :hidden:


Contributing to palma
=====================

Thank you for considering contributing to palma ! This guide will help you get
started with the contribution process.

Forking the Repository
----------------------

To contribute, fork the repository on GitHub. Click the "Fork" button on the top right corner
of the repository page. This creates a copy of the repository in your GitHub account.

::

    git clone https://github.com/your-username/palma
    cd repository

Create a virtual environment and install the project dependencies:

::

    python -m venv venv
    source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
    pip install -r requirements.txt

Create a branch for your changes:

::

    git checkout -b feature-branch

Make your changes and commit:

::

    git add .
    git commit -m "Your descriptive commit message"



Testing with Pytest
--------------------

We use pytest for testing our code. Ensure you have it installed:

::

    pip install pytest

Run the tests using:

::

    pytest tests

Make sure all tests pass before submitting your changes.

Building the Documentation
--------------------------

To build the documentation, ensure you have the necessary documentation tools
installed using the following prompt :

::

    pip install .[doc]

Build the documentation:

::

    cd docs
    sphinx-build source build

This generates the documentation in the ``docs/build`` directory.
Open ``index.html`` in a web browser to review your changes.
Before pushing for ideas make sure you have correctly documented your code !


Making a Merge Request
----------------------

Push your changes to your fork:

::

    git push origin feature-branch

Visit your fork on GitHub and click the "Compare & pull request" button.
Provide a clear and concise description of your changes in the pull request.


