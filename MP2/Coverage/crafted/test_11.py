import pytest
from func_11 import string_xor
def test_equal_length_strings():
    assert string_xor('010', '110') == '100'

def test_unequal_length_strings():
    assert string_xor('010', '1101') == '1001'

def test_all_zeros():
    assert string_xor('000', '000') == '000'

def test_all_ones():
    assert string_xor('111', '111') == '111'

def test_special_cases():
    assert string_xor('01010101', '10101010') == '11111111'
