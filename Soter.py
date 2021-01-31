import random

def perm(n, np):
    p = []
    d = 1
    for i in range(n):
        if np % 2 == 0:
            p.append(d)
            d = 1
        else:
            d += 1
        np //= 2
    return p


def test(ex_n):
    for ex_p in range(2 ** (ex_n - 1)):
        p = perm(ex_n, ex_p)



def randperm(n):
    np = random.randint(0, 2 ** (n - 1))
    return perm(n, np)

print(randperm(10))
