from vpython import *

# Create a scene
scene = canvas(title="Checkerboard Surface", width=800, height=800)

# Define the size of the checkerboard
size = 8  # 8x8 board
square_size = 1  # Size of an individual square
height_difference = (
    0.5  # The difference in height between the raised and normal squares
)

# Loop through the rows and columns to create the checkerboard
for i in range(size):
    for j in range(size):
        # Determine the height of the square based on its position
        if (i + j) % 2 == 0:
            height = height_difference
        else:
            height = 0

        box(
            pos=vector(i * square_size, j * square_size, height / 2),
            size=vector(square_size, square_size, height),
            color=color.white if (i + j) % 2 == 0 else color.black,
        )

# Keep the scene running
while True:
    rate(100)
