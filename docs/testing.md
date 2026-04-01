# Testing

The repository now keeps test and debug scripts under [tests](https://github.sydney.edu.au/CS33-1/CS33-1/tree/main/tests).

Current scripts:

- `python tests/test_llm.py`
  Checks whether the configured LLM endpoint can be called successfully.
- `python tests/test_window4.py`
  Exercises selected helper functions from the manager and business translator modules.

These scripts are currently lightweight validation scripts rather than a full `pytest` test suite.