from math import log2, exp2
from src.gp.tinyverse import Function

def transform_log(x):
    return 0 if x == 0 else log2(x)

def log_add(a,b):
    if a == 0:
        res = b
    elif b == 0:
        res = a
    else:
        res =  max(a,b) + log2(1+exp2(-abs(a-b)))
    return res

def log_mul(a,b):
    if a == 0 or b == 0:
        res = 0
    else:
        res =  a + b
    return res

LOG_ADD = Function(2, "LOG_ADD", log_add)
LOG_MUL = Function(2, "LOG_MUL", log_mul)

