H, M, T = int(input()), int(input()), int(input())

new_M = (M + T) % 60
new_H = (H + (M + T) // 60) % 24

print(f"{new_H:02d}:{new_M:02d}")