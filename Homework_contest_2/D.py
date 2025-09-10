import numpy as np

Height, Width = map(int, input().split())
R_image, G_image, B_image = map(int, input().split())
R_border, G_border, B_border = map(int, input().split())

image = np.full((Height, Width, 3), (R_border, G_border, B_border), dtype=np.uint8)
image[2:-2, 2:-2] = R_image, G_image, B_image

print(image)