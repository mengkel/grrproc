# Script to automate build for PyPI.

rm -fr dist
cd grrproc
black --line-length=79 *.py
cd ..
python -m pip install --upgrade build
python -m build
python -m pip install --upgrade twine
