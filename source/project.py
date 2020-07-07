""" 
    Alexandre Norcia Medeiros    - nUSP: 10295583
    Daniel Penna Chaves Bertazzo - nUSP: 10349561
    Gabriel Citroni Uliana       - nUSP: 9779367 
    
    ICMC - USP, 2020 - 1o semestre

    SCC0251 - Digital Image Processing.
    Final Project - Image Mosaic Collage Composition
"""

# oficial

import numpy as np
import imageio
from skimage.color import rgb2hsv, hsv2rgb
import sys


def setup_progressbar(bar_width=50):
    # setup progressbar
    sys.stdout.write("[%s]" % (" " * bar_width))
    sys.stdout.flush()
    # return to start of line, after '['
    sys.stdout.write("\b" * (bar_width+1))


def print_progress(n, i, bar_width=50):
    # n -> total problem size, i-> iteration
    # 1/100 -> 1 % --> 1/bar_width -> 1% of bar width
    if (i % (round((1.0/bar_width)*n)) == 0):
        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()


def finish_progressbar():
    sys.stdout.write("]\n")  # this ends the progress bar


def normalize_image(img, new_min=0, new_max=255):
    """ returns a image with its values normalized between [new_min:new_max] """

    old_min = np.amin(img)
    old_max = np.amax(img)

    return new_min + (((img - old_min) * (new_max - new_min)) / (old_max - old_min))
    # return (new_max + new_min)/2 # returns a solid color


def downscale(img, n, m):
    """ Takes an image by input and downscale it to the desired resolution """

    out_img = np.zeros((n, m, 3))
    xi, yi = img.shape[0:2]

    if n > xi or m > yi:
        print("Error: Desired resolution larger than original. Downscale not possible.")
        return out_img

    x_factor = xi // n
    y_factor = yi // m

    # get the pixels of the output image
    for x in range(n):
        for y in range(m):
            i = x_factor * x
            j = y_factor * y

            frame = img[i:i+x_factor, j:j+y_factor]
            out_img[x, y, 0] = frame[:, :, 0].sum() / (x_factor * y_factor)
            out_img[x, y, 1] = frame[:, :, 1].sum() / (x_factor * y_factor)
            out_img[x, y, 2] = frame[:, :, 2].sum() / (x_factor * y_factor)

    return out_img


def fix_division(xi, xf, axis_size):
    # make sure that the division have at least 1 pixel and does not surpasses the axis
    xi, xf = np.clip([xi, xf], 0, axis_size)

    if(xf-xi <= 1):
        xi -= 1
        xf += 1

    return np.clip([xi, xf], 0, axis_size)


def find_division(divisions, xi, xf, yi, yf, n_imgs):
    e = [xf - xi, yf - yi]

    found = False
    i = 0
    img = 0
    while(i < n_imgs and not found):
        if e+[i] in divisions:
            found = True
            img = i
        i += 1

    if not found:
        # print("Divisao nova!")
        img = np.random.randint(0, n_imgs)
        divisions.append(e+[img])

    return divisions.index(e+[img])


def create_division_dist(n_elem=100, n_div=10, mean=100, std=25, fix_black=0, axis_size=1000):
    """ generates elements in the normal distribuiton that define the shapes in the mosaic
    """
    rand_x = np.abs(np.random.normal(loc=mean, scale=std, size=n_div))

    u = np.zeros((n_elem))  # mosaic proportion in the axis dimension
    # fill the vectors that will define the mosaic division
    for i in range(n_elem):
        u[i] = rand_x[np.random.randint(0, n_div)]

    # normalize the vectors, so it represents the proportion that each tile occupies in the mosaic
    u = u / np.sum(u)

    # possible black border fix
    if (fix_black == 1): # add the numeric error to the last element, so it can cover all pixels in the image
        missing_x = axis_size - (np.sum((u*axis_size).astype(np.int))) # numeric error
        u[-1] += missing_x / axis_size

    return u


def create_mosaic_division(img, n_imgs=1, method=1, resolution=(50, 50), rand_parameters=(10, 10, 100, 25), fix_black=1):
    """ return an array of index that define a mosaic composition to overlay the image 
    
        input:
            resolution: amount of tiles in the (x,y) dimensions with compose the mosaic
            method: 1 - divides the image in same sized rectangles
                    2 - creates a random mosaic -- low amount of unique division, faster algorithm
                    3 - creates a random mosaic with a more varied y dimension -- medium amount of unique divions
                    4 - creates a random mosaic with more unique division in x and y dimensions -- high amount of unique divisions

            p: used in the 3 method, probability of aglomerating tiles togheter
            shapes: (amount of different shapes that can occour in the x dimension, mean, sd)
        return:
            mosaic: array of tuples defining the subimages range (xi, xf, yi, yf)
            divisions: array of tuples defining the sizes that the tiles need to downscale after

    """

    x, y = img.shape[0:2]

    # mosaic[i] = [xi, xf, yi, yf, d]  
    mosaic = []
    # divisions[i] = [xf - xi, yf - yi, e] e -> 0 = primeira imagem, 1 = segunda imagem ... n = n imagem
    divisions = []

    if method == 1:
        # divides the image in rectagles of the same size

        n = x // resolution[0]
        m = y // resolution[1]

        # number of pixels not being represented in the x and y dimension
        missing_x = 0
        missing_y = 0
        if fix_black == 1:
            missing_x = round(((x/resolution[0]) - (x//resolution[0])) * resolution[0])
            missing_y = round(((y/resolution[1]) - (y//resolution[1])) * resolution[1])
        print("Mosaic divison: n = ", n, ", m = ", m)
        print("Black border size: ", missing_x, missing_y)

        offset_x = 0
        for i in range(resolution[0]):
            xi = i*n + offset_x
            if i < missing_x:
                offset_x += 1

            xf = (i+1)*n + offset_x

            offset_y = 0
            for j in range(resolution[1]):
                yi = j*m + offset_y
                if j < missing_y:
                    offset_y += 1
                yf = (j+1)*m + offset_y

                d = find_division(divisions, xi, xf, yi, yf, n_imgs)
                mosaic.append([xi, xf, yi, yf, d])

    elif method == 2:
        # creates a random mosaic in the x and y dimension, while mantaining the desired resolution

        u = create_division_dist(n_elem=resolution[0], n_div=rand_parameters[0], mean=rand_parameters[2], std=rand_parameters[3], fix_black=fix_black, axis_size=x)
        v = create_division_dist(n_elem=resolution[1], n_div=rand_parameters[1], mean=rand_parameters[2], std=rand_parameters[3], fix_black=fix_black, axis_size=y)

        # fill the mosaic divison
        last_x = 0
        for i in range(resolution[0]):
            xi = last_x
            xf = int(round(u[i]*x + xi))
            xi, xf = fix_division(xi, xf, x)
            last_x = xf

            last_y = 0
            for j in range(resolution[1]):
                yi = last_y
                yf = int(round(v[j]*y + yi))
                yi, yf = fix_division(yi, yf, y)
                last_y = yf


                d = find_division(divisions, xi, xf, yi, yf, n_imgs)
                mosaic.append([xi, xf, yi, yf, d])

    elif method == 3:
        # creates a random mosaic in the x and y dimension, while mantaining the desired resolution

        # unico tipo de linhas, onde todos tiles tem a mesma altura numa mesma linha
        u = create_division_dist(n_elem=resolution[0], n_div=rand_parameters[0],
                                 mean=rand_parameters[2], std=rand_parameters[3], fix_black=fix_black, axis_size=x)

        # fill the mosaic divison
        last_x = 0
        for i in range(resolution[0]):
            xi = last_x
            xf = int(round(u[i]*x + xi))
            xi, xf = fix_division(xi, xf, x)
            last_x = xf

            # varias colunas com larguras diferentes
            v = create_division_dist(n_elem=resolution[1], n_div=rand_parameters[1],
                                            mean=rand_parameters[2], std=rand_parameters[3], fix_black=fix_black, axis_size=y)

            last_y = 0
            for j in range(resolution[1]):
                yi = last_y
                yf = int(round(v[j]*y + yi))
                yi, yf = fix_division(yi, yf, y)
                last_y = yf

                d = find_division(divisions, xi, xf, yi, yf, n_imgs)
                mosaic.append([xi, xf, yi, yf, d])

    elif method == 4:
        # creates a random mosaic in the x and y dimension, while mantaining the desired resolution

        # varias colunas aleatorias com larguras diferentes
        for i in range(resolution[0]):

            v = create_division_dist(n_elem=resolution[1], n_div=rand_parameters[1],
                                     mean=rand_parameters[2], std=rand_parameters[3], fix_black=fix_black, axis_size=y)
            last_y = 0
            for j in range(resolution[1]):
                yi = last_y
                yf = int(round(v[j]*y + yi))
                yi, yf = fix_division(yi, yf, y)
                last_y = yf
                
                # mosaic.append([xi=0, xf=0, yi, yf, d=0])
                mosaic.append([0, 0, yi, yf, 0])

        # varias linhas aleatorias com alturas diferentes
        for j in range(resolution[1]):

            u = create_division_dist(n_elem=resolution[0], n_div=rand_parameters[0],
                                     mean=rand_parameters[2], std=rand_parameters[3], fix_black=fix_black, axis_size=x)
            last_x = 0
            for i in range(resolution[0]):
                xi = last_x
                xf = int(round(u[i]*x + xi))
                xi, xf = fix_division(xi, xf, x)
                last_x = xf

                yi, yf = mosaic[(i*resolution[1]) + j][2:4]
                mosaic[(i*resolution[1]) + j] = [xi, xf, yi, yf, 0]
        # acha e completa os buraco

        for i in range(resolution[0]):
            for j in range(resolution[1]):
                xi, xf, yi, yf, d = mosaic[(i*resolution[1]) + j]

                d = find_division(divisions, xi, xf, yi, yf, n_imgs)
                mosaic[(i*resolution[1]) + j] = [xi, xf, yi, yf, d]


    return mosaic, divisions


def get_predominant_color(img):
    """ Returns the predominant color present in the image and the max and min values
            It defines the predominant color using the mean of all the 3 channels
    """

    color = np.zeros((3))
    max_colors = np.zeros((3))
    min_colors = np.zeros((3))

    max_colors[0] = np.max(img[:, :, 0])
    max_colors[1] = np.max(img[:, :, 1])
    max_colors[2] = np.max(img[:, :, 2])

    min_colors[0] = np.min(img[:, :, 0])
    min_colors[1] = np.min(img[:, :, 1])
    min_colors[2] = np.min(img[:, :, 2])

    # mean of all colors in the image
    color[0] = np.mean(img[:, :, 0])
    color[1] = np.mean(img[:, :, 1])
    color[2] = np.mean(img[:, :, 2])


    return color, min_colors, max_colors


def alter_color(img, predominant_color, min_color, max_color, method=1):
    """ alters the color of the image to approximate it to the predominant_color parameter
        
        inputs:
            img:
            predominant_color:
            min_color:
            max_color:
            method:
    """

    out_img = np.array(img, copy=True)

    if method == 1:

        out_img[:, :, 0] = out_img[:, :, 0] * \
            (predominant_color[0] / np.sum(predominant_color))  # R
        out_img[:, :, 1] = out_img[:, :, 1] * \
            (predominant_color[1] / np.sum(predominant_color))  # G
        out_img[:, :, 2] = out_img[:, :, 2] * \
            (predominant_color[2] / np.sum(predominant_color))  # B

    elif method == 2:

        # transforms to the HSV color representation
        out_img = rgb2hsv(out_img)
        predominant_color = rgb2hsv(np.reshape(predominant_color, (1, 1, 3)))

        out_img[:, :, 0] = (predominant_color[:, :, 0])  # alters hue
        out_img[:, :, 1] = (predominant_color[:, :, 1])  # alters saturation

        out_img = hsv2rgb(out_img)

    # returns to original color interval in rgb to accurately represent the intensity (black/white)
    out_img = normalize_image(out_img, new_min=min_color, new_max=max_color)

    return np.clip(out_img, 0, 255)


def mosaic_transform(canvas_img, tile_imgs, n_imgs=1, division_method=1, resolution=(50, 50),
                     rand_parameters=(10, 10, 100, 25), fix_black=1, alter_color_method=2):
    """ creates a mosaic that represents one image using other images
        
        inputs:
            canvas_img: image to be generated with the mosaic 
            tile_imgs: images used as mosaic tiles
            division_method: 1 - every tile have the same size
                             2 - tiles with different sizes

            resolution: number of tiles to compose the final image in each dimension

            get_color_method: 1 - Obtain predominant color by the mean of all pixels
                              2 - 

            alter_color_method: 1 - RGB color processing
                                2 - HSV color processing
    """

    out_img = np.zeros(canvas_img.shape)  # creates output image

    print("Creating mosaic...")
    mosaic, divisions = create_mosaic_division(canvas_img, n_imgs=n_imgs,
        method=division_method, resolution=resolution, rand_parameters=rand_parameters, fix_black=fix_black)

    n_div = len(divisions)
    print("number of unique divison: ", n_div, ", total divisions: ", len(mosaic))
    # print(divisions)

    print("Creating tile images...")
    setup_progressbar()
    i = 0

    down_tiles = []
    for division in divisions:
        down_tiles.append(downscale(tile_imgs[division[2]], division[0], division[1]))
        i += 1
        print_progress(n=n_div, i=i)
    finish_progressbar()

    print("Composing the Mosaic...")
    setup_progressbar()
    i = 0

    for subimage in mosaic:
        xi, xf, yi, yf, d = subimage

        predominant_color, min_color, max_color = get_predominant_color(
            canvas_img[xi:xf, yi:yf])
        tile = alter_color(down_tiles[d], predominant_color, min_color, max_color, method=alter_color_method)
        out_img[xi:xf, yi:yf] = tile
        i += 1
        print_progress(n=resolution[0]*resolution[1], i=i)
    finish_progressbar()

    return out_img


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

if division_method > 1:
    print("Enter the amount of unique elements in the x and y dimensions: ")
    ux = int(input())
    uy = int(input())
    print("Enter the normal distribuition parameters (mean and standard deviation): ")
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

# reads image from directory and changes the data type
canvas_img = imageio.imread('../images/'+img_name_canvas).astype(np.float)
tile_imgs = []
for name in tile_names:
    tile_imgs.append(imageio.imread('../images/'+name).astype(np.float))


out_img = mosaic_transform(canvas_img, tile_imgs, n_imgs=n_imgs, division_method=division_method, 
    resolution=(x, y), rand_parameters=(ux, uy, mean, std), fix_black=fix_black, alter_color_method=alter_color_method)
out_img = out_img.astype(np.uint8)

imageio.imwrite('../output_images/'+out_name, out_img.astype(np.uint8))

print("Acabou")
