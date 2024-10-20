import pytest
from func_139 import special_factorial
def test_special_factorial():
    assert special_factorial(0) == 1
    assert special_factorial(1) == 1
    assert special_factorial(2) == 4
    assert special_factorial(3) == 288
    assert special_factorial(4) == 4800
    assert special_factorial(5) == 1036800

