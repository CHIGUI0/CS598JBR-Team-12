import pytest
from func_124 import valid_date
def test_correct_format():
    assert valid_date('03-11-2000') == True

def test_invalid_month():
    assert valid_date('15-01-2012') == False

def test_invalid_day():
    assert valid_date('04-0-2040') == False

def test_valid_date():
    assert valid_date('06-04-2020') == True

def test_invalid_format():
    assert valid_date('06/04/2020') == False

def test_leap_year():
    assert valid_date('02-29-2000') == True

def test_non_leap_year():
    assert valid_date('02-29-2001') == False

def test_smallest_date():
    assert valid_date('01-01-1900') == True

def test_largest_date():
    assert valid_date('12-31-2099') == True

def test_day_zero():
    assert valid_date('01-00-2000') == False

def test_day_thirty_one():
    assert valid_date('02-30-2001') == False

def test_day_thirty():
    assert valid_date('04-30-2000') == False

def test_day_thirty_and_a_half():
    assert valid_date('06-30-2000') == False

def test_day_thirty_one_and_a_half():
    assert valid_date('07-31-2000') == False

def test_day_thirty_two():
    assert valid_date('08-32-2000') == False

def test_day_thirty_three():
    assert valid_date('09-33-2000') == False

def test_day_thirty_four():
    assert valid_date('10-34-2000') == False

def test_day_thirty_five():
    assert valid_date('11-35-2000') == False

def test_day_thirty_six():
    assert valid_date('12-36-2000') == False
