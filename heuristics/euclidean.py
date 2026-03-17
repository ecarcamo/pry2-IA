import math


def euclidean(a, b):

    r1, c1 = a
    r2, c2 = b

    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)