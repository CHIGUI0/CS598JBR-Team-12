import pytest
from func_17 import parse_music
def test_empty_string():
    assert parse_music("") == []

def test_whole_notes():
    assert parse_music("o o o o") == [4, 4, 4, 4]

def test_half_notes():
    assert parse_music("o| o| o| o|") == [2, 2, 2, 2, 2]

def test_quater_notes():
    assert parse_music(".| .| .| .| .| .| .| .| .|") == [1]*9

def test_mixed_notes():
    assert parse_music("o o| .| o| o| .| .| .| .| o o") == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]

def test_boundary_whole_notes():
    assert parse_music("o") == [4]

def test_boundary_half_notes():
    assert parse_music("o|") == [2]

def test_boundary_quater_notes():
    assert parse_music(".|") == [1]
