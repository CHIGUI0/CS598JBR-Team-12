def test_add():
    assert add(2, 3) == 5
    assert add(5, 7) == 12
    assert add(0, 0) == 0
    assert add(-2, 3) == 1
    assert add(100, 0) == 100
