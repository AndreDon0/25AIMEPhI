def cashe(func):
    stored_results = {}

    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in stored_results:
            stored_results[key] = func(*args, **kwargs)
        return stored_results[key]

    return wrapper

@cashe
def A(n, m):
    import sys
    sys.setrecursionlimit(1000000000)

    if n == 0: 
        return m + 1
    elif m == 0:
        return A(n - 1, 1)
    else:
        return A(n - 1, A(n, m - 1))

n, m = map(int, input().split())
print(A(n, m))