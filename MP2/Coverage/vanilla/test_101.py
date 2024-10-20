import pytest
from func_101 import words_string  # replace 'your_module' with the actual module name
def test_words_string():
    assert words_string("Hi, my name is John") == ["Hi", "my", "name", "is", "John"]
    assert words_string("One, two, three, four, five, six") == ["One", "two", "three", "four", "five", "six"]
    assert words_string("") == []
    assert words_string(None) == []

