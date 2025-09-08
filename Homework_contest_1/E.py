def staples_extermination(s: str) -> str:
    import re

    w = re.compile(r"[^\(\)\[\]\{\}\<\>]+")
    s = w.sub('', s)

    p = re.compile(r"\(\)|\[\]|\{\}|\<\>")

    r = p.sub('', s)
    while s != r:
        s = r
        r = p.sub('', s)
    
    return s


print("YES" if staples_extermination(input()) == '' else "NO")
