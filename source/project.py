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


def create_mosaic_division(img, method=1, resolution=(50, 50), p=0.7, random_shapes=(4, 4, 100, 25)):
    """ return an array of index that define a mosaic composition to overlay the image 
    
        input:
            resolution: amount of tiles in the (x,y) dimensions with compose the mosaic
            method: 1 - divides the image in rectagles of the same size
                    2 - creates a random mosaic using a normal distribuiton in the x and y dimension, while mantaining the desired resolution
                    3 -
            p: used in the 3 method, probability of aglomerating tiles togheter
            shapes: (amount of different shapes that can occour in the x dimension, mean, sd)
        return:
            mosaic: array of tuples defining the subimages range (xi, xf, yi, yf)
            divisions: array of tuples defining the sizes that the tiles need to downscale after

        
        130 / 13 = 10 + resto

        [ 2 2 2 2 2 1 1 1]

        x = 3168 y = 4752  resolution=(100, 100)
        3168 / n = 100 + borda_preta
        n = 3168 / 100 = 31.68 = 31 + 0.68

        100 * 0.68 = 68 >> 31
        qts imgs * borda = nro de imagens com pixel a mais

        0 - 67 imagens 32 pixels
        68 - 99 -> 31 pixels

        3168 / 8

        4752 / m = 100 + borda
        m = 4752 / 100 = 47.52 = 47 + 0.52

        divisoes = [[31, 47], [31, 48], [32, 47], [32, 48]]
        mosaico[i] = [xi=100, xf=131, 100, 147, divisao=0, imagem=0]
        mosaico[-1] = 0 -> divisoes[0] = 31x47

    """

    x, y = img.shape[0:2]

    n = x // resolution[0]
    m = y // resolution[1]

    # number of pixels not being represented in the x dimension
    missing_x = round(((x/resolution[0]) - (x//resolution[0])) * resolution[0])
    # number of pixels not being represented in the y dimension
    missing_y = round(((y/resolution[1]) - (y//resolution[1])) * resolution[1])

    print(missing_x, missing_y)

    mosaic = np.zeros((resolution[0]*resolution[1], 5), dtype=int)
    divisions = []

    if method == 1:
        # divides the image in rectagles of the same size
        print("Mosaic divison: n = ", n, ", m = ", m)

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

                # xi = i*n 
                # yi = j*m 
                # xf = (i+1)*n 
                # yf = (j+1)*m 

                e = [xf - xi, yf - yi]
                if e not in divisions:
                    print("Divisao nova!")
                    divisions.append(e)

                d = divisions.index(e)

                mosaic[(i*resolution[1]) + j] = [xi, xf, yi, yf, d]

    elif method == 2:
        # creates a random mosaic in the x and y dimension, while mantaining the desired resolution

        # generates elements in the normal distribuiton that define the shapes in the mosaic
        rand_x = np.random.normal(loc=random_shapes[2], scale=random_shapes[3], size=random_shapes[0])
        rand_y = np.random.normal(loc=random_shapes[2], scale=random_shapes[3], size=random_shapes[1])

        u = np.zeros((resolution[0]))  # mosaic proportion in the x dimension
        v = np.zeros((resolution[1]))  # mosaic proportion in the y dimension
        # fill the vectors that will define the mosaic division
        for i in range(0, n):
            u[i] = rand_x[np.random.randint(0, random_shapes[0])]
        for i in range(0, m):
            v[i] = rand_y[np.random.randint(0, random_shapes[1])]

        # normalize the vectors, so it represents the proportion that each tile occupies in the mosaic
        u = u / np.sum(u)
        v = v / np.sum(v)

        # fill the mosaic divison
        for i in u:
            for j in v:
                i, j

    print(offset_x, offset_y)

    return mosaic, divisions


def get_predominant_color(img, method=1):
    """ returns the predominant color present in the image
        
        inputs:
            method: 1 - mean of all pixels
                    2 - color histogram (with margin?) (200, 200, 200) -- (199, 200, 199)
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

    if method == 1:
        # mean of all colors in the image
        color[0] = np.mean(img[:, :, 0])
        color[1] = np.mean(img[:, :, 1])
        color[2] = np.mean(img[:, :, 2])

    # elif method == 2:

    #     color[0] = np.argmax(np.bincount(img[ :, :, 0]))
    #     color[1] = np.argmax(np.bincount(img[ :, :, 1]))
    #     color[2] = np.argmax(np.bincount(img[ :, :, 2]))

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

        out_img[:, :, 0] = out_img[:, :, 0] * (predominant_color[0] / np.sum(predominant_color))  # R
        out_img[:, :, 1] = out_img[:, :, 1] * (predominant_color[1] / np.sum(predominant_color))  # G
        out_img[:, :, 2] = out_img[:, :, 2] * (predominant_color[2] / np.sum(predominant_color))  # B

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


def mosaic_transform(canvas_img, tile_img, division_method=1, resolution=(8, 8), get_color_method=1, alter_color_method=1):
    """ creates a mosaic that represents one image using other images
        
        inputs:
            canvas_img: image to be generated with the mosaic 
            tile_img: image used as mosaic tiles
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
    mosaic, divisions = create_mosaic_division(
        canvas_img, method=division_method, resolution=resolution)

    print(len(divisions))
    print(divisions)

    print("Creating tile images...")
    down_tiles = []
    for division in divisions:
        down_tiles.append(downscale(tile_img, division[0], division[1]))

    print(len(down_tiles))

    print("Composing the Mosaic...")
    for subimage in mosaic:
        xi, xf, yi, yf, d = subimage

        predominant_color, min_color, max_color = get_predominant_color(canvas_img[xi:xf, yi:yf], method=get_color_method)
        tile = alter_color(down_tiles[d], predominant_color, min_color, max_color, method=alter_color_method)
        out_img[xi:xf, yi:yf] = tile

    return out_img


# reads inputs
img_name_canvas = str(input()).rstrip()
img_name_tile = str(input()).rstrip()
save = 1

# reads image from directory
canvas_img = imageio.imread('../images/'+img_name_canvas)
tile_img = imageio.imread('../images/'+img_name_tile)

# change data type
canvas_img = canvas_img.astype(np.float)
tile_img = tile_img.astype(np.float)

out_img = mosaic_transform(canvas_img, tile_img, resolution=(100, 100))
out_img = out_img.astype(np.uint8)

# verifies if the modified image needs to be saved
if (save):
    imageio.imwrite('../output_images/output_image.jpg', out_img.astype(np.uint8))

print("Acabou")
