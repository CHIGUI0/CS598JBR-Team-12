import pytest
from func_74 import total_match
def test_empty_lists():
    assert total_match([], []) == []

def test_single_string_lists():
    assert total_match(['hi'], ['hI']) == ['hI']

def test_multiple_string_lists_different_lengths():
    assert total_match(['hi', 'admin'], ['hI', 'Hi']) == ['hI', 'Hi']

def test_multiple_string_lists_equal_lengths():
    assert total_match(['hi', 'admin'], ['hi', 'Hi']) == ['hi', 'Hi']

def test_boundary_checks():
    assert total_match(['hi'], ['hI', 'Hi']) == ['hI', 'Hi']
    assert total_match(['hi', 'admin'], ['hi']) == ['hi']

def test_various_lengths_and_strings():
    assert total_match(['hi', 'admin'], ['hI', 'Hi', 'admin']) == ['hI', 'Hi', 'admin']
    assert total_match(['hi'], ['hI', 'Hi', 'admin', 'project']) == ['hI', 'Hi', 'admin', 'project']

