# Build and publish to PyPI
python3 -m pip install --upgrade build twine hatch
hatch build
# To test upload to TestPyPI:
# twine upload --repository testpypi dist/*
# To upload to PyPI:
# twine upload dist/*
