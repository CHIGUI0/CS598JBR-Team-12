import pytest
from func_11 import string_xor  # replace 'your_module' with the actual module name
def test_string_xor():
    assert string_xor('010', '110') == '100'
    assert string_xor('101', '001') == '100'
    assert string_xor('111', '111') == '000'
    assert string_xor('000', '000') == '000'

