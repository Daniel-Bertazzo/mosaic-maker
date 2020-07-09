from project import *

# reads inputs
print("Enter the image name that will be represented as a mosaic: ")
img_name_canvas = str(input()).rstrip()

print("Enter how many images do you want to compose the final image: ")
n_imgs = int(input())

print("Enter the names of all the tile images: ")
tile_names = []
for i in range(n_imgs):
    tile_names.append(str(input()).rstrip())

print("Enter the mosaic resolution (number of tiles composing the mosaic): ")
x = int(input())
y = int(input())

print("Enter the division method to compose the mosaic: ")
print("1 - divides the image in same sized rectangles")
print("2 - creates a random mosaic                               -- faster algorithm, low amount of unique division")
print("3 - creates a random mosaic with a more varied y dimension                  -- medium amount of unique divions")
print("4 - creates a random mosaic with more unique division in x and y dimensions -- high amount of unique divisions\n")
division_method = int(input())

ux = 0
uy = 0
mean = 0
std = 0
if division_method > 1:
    print("Enter the amount of unique elements in the x and y dimensions: ")
    ux = int(input())
    uy = int(input())
    print("Enter the normal distribution parameters (mean and standard deviation): ")
    mean = float(input())
    std = float(input())

print("Enter if you want to fix black borders (1 - true, 0 - false)")
fix_black = int(input())

print("Enter the color method to be applied when altering the tiles images colors")
print("1 - RGB color processing")
print("2 - HSV color processing\n")
alter_color_method = int(input())

print("Enter the output image name with the format (ex: out_image.jpg): ")
out_name = str(input()).rstrip()

# reads image from directory and changes the data type, also it ignores the alpha channel in png format
canvas_img = imageio.imread(
    '../images/'+img_name_canvas)[:, :, 0:3].astype(np.float)
tile_imgs = []
for name in tile_names:
    tile_imgs.append(imageio.imread('../images/'+name)
                     [:, :, 0:3].astype(np.float))

# creates mosaic
out_img = mosaic_transform(canvas_img, tile_imgs, n_imgs=n_imgs, resolution=(x, y), division_method=division_method,
                           rand_parameters=(ux, uy, mean, std), fix_black=fix_black, alter_color_method=alter_color_method)
out_img = out_img.astype(np.uint8)

# save image
imageio.imwrite('../output_images/'+out_name, out_img.astype(np.uint8))

print("Finished.")

# reads inputs
print("Enter the image name that will be represented as a mosaic: ")
img_name_canvas = str(input()).rstrip()

print("Enter how many images do you want to compose the final image: ")
n_imgs = int(input())

print("Enter the names of all the tile images: ")
tile_names = []
for i in range(n_imgs):
    tile_names.append(str(input()).rstrip())

print("Enter the mosaic resolution (number of tiles composing the mosaic): ")
x = int(input())
y = int(input())

print("Enter the division method to compose the mosaic: ")
print("1 - divides the image in same sized rectangles")
print("2 - creates a random mosaic                               -- faster algorithm, low amount of unique division")
print("3 - creates a random mosaic with a more varied y dimension                  -- medium amount of unique divions")
print("4 - creates a random mosaic with more unique division in x and y dimensions -- high amount of unique divisions\n")
division_method = int(input())

ux = 0
uy = 0
mean = 0
std = 0
if division_method > 1:
    print("Enter the amount of unique elements in the x and y dimensions: ")
    ux = int(input())
    uy = int(input())
    print("Enter the normal distribution parameters (mean and standard deviation): ")
    mean = float(input())
    std = float(input())

print("Enter if you want to fix black borders (1 - true, 0 - false)")
fix_black = int(input())

print("Enter the color method to be applied when altering the tiles images colors")
print("1 - RGB color processing")
print("2 - HSV color processing\n")
alter_color_method = int(input())

print("Enter the output image name with the format (ex: out_image.jpg): ")
out_name = str(input()).rstrip()

# reads image from directory and changes the data type, also it ignores the alpha channel in png format
canvas_img = imageio.imread('../images/'+img_name_canvas)[:, :, 0:3].astype(np.float)
tile_imgs = []
for name in tile_names:
    tile_imgs.append(imageio.imread('../images/'+name)[:, :, 0:3].astype(np.float))

# creates mosaic
out_img = mosaic_transform(canvas_img, tile_imgs, n_imgs=n_imgs, resolution=(x, y), division_method=division_method,
                           rand_parameters=(ux, uy, mean, std), fix_black=fix_black, alter_color_method=alter_color_method)
out_img = out_img.astype(np.uint8)

# save image
imageio.imwrite('../output_images/'+out_name, out_img.astype(np.uint8))

print("Finished.")