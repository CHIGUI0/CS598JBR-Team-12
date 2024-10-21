import pytest
from func_112 import reverse_delete
def test_empty_string():
    assert reverse_delete("", "") == ("", True)

def test_no_deletion():
    assert reverse_delete("abc", "") == ("abc", False)

def test_all_deletion():
    assert reverse_delete("abc", "abc") == ("", True)

def test_palindrome_check():
    assert reverse_delete("abcba", "") == ("abcba", True)

def test_special_characters():
    assert reverse_delete("!@#$%^&*", "") == ("!@#$%^&*", True)

def test_mixed_case_characters():
    assert reverse_delete("AbCdEfGh", "") == ("AbCdEfGh", False)

def test_boundary_checks():
    assert reverse_delete("a", "") == ("a", False)
    assert reverse_delete("aa", "a") == ("", True)
    assert reverse_delete("aaa", "a") == ("a", False)

def test_various_lengths_and_characters():
    assert reverse_delete("abc", "a") == ("bc", False)
    assert reverse_delete("abcabc", "abc") == ("", True)
    assert reverse_delete("abcabcabc", "abc") == ("abc", False)
