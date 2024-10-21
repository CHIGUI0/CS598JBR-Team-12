import pytest
from func_44 import change_base
def test_zero():
    assert change_base(0, 2) == '0'

def test_single_digit_numbers():
    for i in range(1, 10):
        assert change_base(i, 2) == bin(i)[2:]

def test_multiple_digit_numbers():
    assert change_base(123456, 2) == bin(123456)[2:]

def test_special_cases():
    with pytest.raises(ValueError):
        change_base(-1, 2)
    with pytest.raises(ValueError):
        change_base(123456, 1)

def test_base_less_than_10():
    assert change_base(8, 3) == '22'
    assert change_base(7, 2) == '111'

