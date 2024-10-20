import pytest
from func_112 import reverse_delete  # replace 'your_module' with the name of your module
def test_reverse_delete():
    assert reverse_delete("abcde", "ae") == ("bcd", False)
    assert reverse_delete("abcdef", "b") == ("acdef", False)
    assert reverse_delete("abcdedcba", "ab") == ("cdedc", True)

