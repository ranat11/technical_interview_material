# Homework for Technical Interview

This directory is to be used as a "homework" in technical interview for intern/employee candidate.
When being passed to the candidate, only this directory should be passed, and no git history (`.git/`) should exist there.

## Task for candidate

1. Open the existing notebook (`notebook/regression.ipynb`), and understand what is done there.
   - If you want, you can modify it and run the notebook for better understanding. But you will need a few libraries such as `pandas`. You will need to prepare such environment. If you are familiar with a tool [`poetry`](https://python-poetry.org/), you can use the existing setting file `pyproject.toml`/`poetry.lock`.
2. Make this directory a git repository so that all your working history will be recorded and version-controlled from this step.
3. Create a python library in `regression/` and an application code (`main.py`) so that they do the same tasks as in the notebook (`notebook/regression.ipynb`).
   - The tasks include:
     - training regression model
     - testing regression model
   - Each method should be implemented in the library so that they can be reused and unittested.
   - The application code use the methods in the library by importing them.
   - If you used some libraries (Actually I am sure that you will), you need to list all the used libraries and their versions as text file in some formats, so that Arithmer can run the code:
     - `requirements.txt`
     - `poetry.lock`
     - Some other format which are well-used
   - [Bonus] Following [PEP8](https://www.python.org/dev/peps/pep-0008/) is recommended.
   - [Bonus] Adding [docstring](https://realpython.com/documenting-python-code/)/[type annotation](https://realpython.com/python-type-checking/) is recommended in implementing methods in the library.
4. [Bonus] Write unittest in `tests/` for the library (`regression/`).

In 3. and 4., make sure that you make git commits with proper granuality.

### Time budget

The candidate is asked to submit her/his homeworks as soon as it is done, and up to one month.

## Task for interviewers

Once the candidate submits the code, the interviewers should review it for discussion session.
Here are a few remarks:

- This "homework" is not for strict evaluation, but for preparing material for discussion on concrete codebase. Therefore, one does not need to very well-defined/quantitative evaluation in reviewing the code.
- Prepare what to ask/discuss with the candidate at the discussion session.
- Make sure that you give feedback to the candidate regarding the coding style, etc., irrespective of the result.

## Refs

### software engineering

- [git](https://git-scm.com/docs/gittutorial)
- [concept of unittest](https://en.wikipedia.org/wiki/Unit_testing)

### python

- Python:
  - [realpython](https://realpython.com/)
- [Python module](https://realpython.com/python-modules-packages/)
- Installing/managing python environment:
  - [pip](https://realpython.com/what-is-pip/)
  - [poetry](https://realpython.com/effective-python-environment/)
- [unittest with pytest](https://realpython.com/pytest-python-testing/)

### data science/machine learning

- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [Jupyter notebook/lab](https://jupyterlab.readthedocs.io/en/stable/)
