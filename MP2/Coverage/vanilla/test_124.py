import pytest
from func_124 import valid_date  # replace 'your_module' with the name of your module
def test_valid_date():
    assert valid_date('03-11-2000') == True
    assert valid_date('15-01-22012') == False
    assert valid_date('04-0-2040') == False
    assert valid_date('06-04-2020') == True
    assert valid_date('06/04/2020') == False

