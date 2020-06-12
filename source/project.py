import numpy as np
import imageio


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
            # out_img[x, y] = np.sum(np.sum(frame, axis = 1), axis = 0) / (x_factor * y_factor)
            out_img[x, y, 0] = frame[:, :, 0].sum() / (x_factor * y_factor)
            out_img[x, y, 1] = frame[:, :, 1].sum() / (x_factor * y_factor)
            out_img[x, y, 2] = frame[:, :, 2].sum() / (x_factor * y_factor)

    return out_img.astype(np.uint8)


def create_mosaic_division(img, method=1, resolution=(8, 8)):
    """ return an array of index that define a mosaic composition to overlay the image 
    
        input:

        return:
            mosaic: array of tuples defining the subimages range (xi, xf, yi, yf)
    """

    x, y = img.shape[0:2]
    mosaic = None

    if method == 1:
        # divides the image in rectagles of the same size
        n = x // resolution[0]
        m = y // resolution[1]

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

    if method == 1:
        # mean of all colors in the image
        color[0] = np.mean(img[:, :, 0])
        color[1] = np.mean(img[:, :, 1])
        color[2] = np.mean(img[:, :, 2])

    return color


def alter_color(img, predominant_color):
    """ alters the color of the image to approximate it to the predominant_color parameter
        
        inputs:
    """

    # out_img = img ** 2.2 # transform srgb to linear srgb
    # out_img = img
    # out_img = np.log(img + 1)
    out_img = np.array(img, copy=True)
    # out_img = np.array(img, copy=True) ** 2.2  # transform srgb to linear srgb

    # (100, 150, 200) -> (100/(100 + 150 + 200), , ) color example
    out_img[:, :, 0] = out_img[:, :, 0] * (predominant_color[0] / np.sum(predominant_color))
    out_img[:, :, 1] = out_img[:, :, 1] * (predominant_color[1] / np.sum(predominant_color))
    out_img[:, :, 2] = out_img[:, :, 2] * (predominant_color[2] / np.sum(predominant_color))

    # out_img += 50

    # out_img = out_img ** (1/2.2) # transform back to srgb
    # out_img = np.exp(out_img)

    return np.clip(out_img, 0, 255)
    # return out_img


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

    xi, xf, yi, yf = mosaic_division[0]
    down_tile = downscale(tile_img, xf - xi, yf - yi)

    print(mosaic_division)

    count = 0
    for subimage in mosaic_division:
        xi, xf, yi, yf = subimage

        count += 1
        print("subimage: ", count)

        predominant_color = get_predominant_color(
            canvas_img[xi:xf, yi:yf], method=color_method)
        # tile = downscale(tile_img, xf - xi, yf - yi)
        tile = alter_color(down_tile, predominant_color)
        out_img[xi:xf, yi:yf] = tile

    print("out image: ", canvas_img.shape)
    print("mosaic division: ", mosaic_division[-1])

    return out_img


# reads inputs
img_name_canvas = str(input()).rstrip()
img_name_tile = str(input()).rstrip()
save = 1

# reads image from '.' directory
canvas_img = imageio.imread(img_name_canvas)
tile_img = imageio.imread(img_name_tile)

canvas_img = canvas_img.astype(np.float)
tile_img = tile_img.astype(np.float)

out_img = mosaic_transform(canvas_img, tile_img, resolution=(100, 100))

# verifies if the modified image needs to be saved
if (save):
    imageio.imwrite('output images/output_image.png', out_img.astype(np.uint8))

print("Acabou")
