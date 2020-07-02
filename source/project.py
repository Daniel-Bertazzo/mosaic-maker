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


def create_mosaic_division(img, method=1, resolution=(8, 8)):
    """ return an array of index that define a mosaic composition to overlay the image 
    
        input:

        return:
            mosaic: array of tuples defining the subimages range (xi, xf, yi, yf)

        
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

        divisoes = [31x47, 31x48, 32x47, 32x48]
        mosaico[i] = [xi=100, xf=131, 100, 147, 0]
        mosaico[-1] = 0 -> divisoes[0] = 31x47

    """

    x, y = img.shape[0:2]
    mosaic = None

    if method == 1:
        # divides the image in rectagles of the same size
        n = x // resolution[0]
        m = y // resolution[1]

        print("Mosaic divison: n = ", n, ", m = ", m)

        mosaic = np.zeros((resolution[0]*resolution[1], 4), dtype=int)

        for i in range(resolution[0]):
            for j in range(resolution[1]):
                mosaic[(i*resolution[1]) + j] = [i*n, (i+1)*n, j*m, (j+1)*m]

    # elif method == 2:
    #     mosaic = variado

    return mosaic


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

    return color, min_colors, max_colors


def alter_color(img, predominant_color, min_color, max_color):
    """ alters the color of the image to approximate it to the predominant_color parameter
        
        inputs:
    """

    out_img = np.array(img, copy=True)
    out_img = normalize_image(out_img, new_min=0, new_max=1)

    # (100, 150, 200) -> (100/(100 + 150 + 200), , ) color example
    out_img[:, :, 0] = out_img[:, :, 0] * (predominant_color[0] / np.sum(predominant_color))
    out_img[:, :, 1] = out_img[:, :, 1] * (predominant_color[1] / np.sum(predominant_color))
    out_img[:, :, 2] = out_img[:, :, 2] * (predominant_color[2] / np.sum(predominant_color))

    out_img = normalize_image(out_img, new_min=min_color, new_max=max_color)
    return np.clip(out_img, 0, 255)


def mosaic_transform(canvas_img, tile_img, division_method=1, resolution=(8, 8), color_method=1):
    """ creates a mosaic that represents one image using other images
        
        inputs:
            canvas_img: image to be generated with the mosaic 
            tile_img: image used as mosaic tiles
            division_method: 1 - every tile have the same size
                             2 - tiles with different sizes
            resolution: 
            color_method: 
    """

    mosaic_division = create_mosaic_division(
        canvas_img, method=division_method, resolution=resolution)
    out_img = np.zeros(canvas_img.shape)

    print(canvas_img.shape)

    xi, xf, yi, yf = mosaic_division[0]
    down_tile = downscale(tile_img, xf - xi, yf - yi)

    # print(mosaic_division)

    count = 0
    for subimage in mosaic_division:
        xi, xf, yi, yf = subimage

        count += 1
        # print("subimage: ", count)

        predominant_color, min_color, max_color = get_predominant_color(canvas_img[xi:xf, yi:yf], method=color_method)
        # tile = downscale(tile_img, xf - xi, yf - yi) 
        tile = alter_color(down_tile, predominant_color, min_color, max_color)
        out_img[xi:xf, yi:yf] = tile

    print("out image: ", canvas_img.shape)
    print("mosaic division: ", mosaic_division[-1])

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
    imageio.imwrite('../output images/output_image.jpg', out_img.astype(np.uint8))

print("Acabou")
