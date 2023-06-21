def intersection(lst1, lst2):
    """
    返回 lst1 和 lst2 的交集；
    返回的交集如果存在多个，以 lst1 的顺序排列
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
