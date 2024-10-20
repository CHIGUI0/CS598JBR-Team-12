import pytest
from func_144 import simplify  # replace 'your_module' with the name of your module
def test_simplify():
    assert simplify("1/5", "5/1") == True
    assert simplify("1/6", "2/1") == False
    assert simplify("7/10", "10/2") == False

