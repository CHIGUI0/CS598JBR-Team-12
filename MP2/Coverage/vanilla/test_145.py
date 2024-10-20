import pytest
from func_145 import order_by_points  # replace 'your_module' with the actual module name
def test_order_by_points():
    assert order_by_points([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11]
    assert order_by_points([]) == []
    assert order_by_points([15, 25, 35]) == [15, 35, 25]  # test with different numbers
    assert order_by_points([-15, -25, -35]) == [-15, -35, -25]  # test with negative numbers

