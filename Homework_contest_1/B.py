def parse_salary(s: str):
    import re

    def empty2none(s):
        return None if s == '' else s
    
    s = s.replace(' ', '')

    pattern = re.compile(
        r'^Подоговорённости()()()$ \
        |^от([\d]+)()(\S+)$ \
        |^до()([\d]+)(\S+)$ \
        |^([\d]+)-([\d]+)(\S+)$'
    )

    m = pattern.match(s)
    gd = m.groups()

    for i in range(0, 12, 3):
        if gd[i] != None:
            return empty2none(gd[i]), \
                   empty2none(gd[i + 1]), \
                   empty2none(gd[i + 2])

    return None, None, None


s = input()

sal_min, sal_max, sal_cur = parse_salary(s)

print("sal_min:", sal_min)
print("sal_max:", sal_max)
print("sal_cur:", sal_cur)
