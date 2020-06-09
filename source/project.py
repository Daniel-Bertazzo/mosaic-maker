import numpy as np
import imageio

def downscale(img, xf, yf):
    """ Takes an image by input and downscale it to the desired resolution """
    
    out_img = np.zeros((xf, yf, 3))
    xi, yi = img.shape[0:2]

    if xf > xi or yf > yi:
        print("Error: Desired resolution larger than original. Downscale not possible.")
        return out_img

    x_factor = xi // xf
    y_factor = yi // yf

    # get the pixels of the output image
    for x in range(xf):
        for y in range(yf):
            i = x_factor * x
            j = y_factor * y

            frame = img[i:i+x_factor, j:j+y_factor]
            # out_img[x, y] = np.sum(np.sum(frame, axis = 1), axis = 0) / (x_factor * y_factor)
            out_img[x, y, 0] = frame[:, :, 0].sum() / (x_factor * y_factor)
            out_img[x, y, 1] = frame[:, :, 1].sum() / (x_factor * y_factor)
            out_img[x, y, 2] = frame[:, :, 2].sum() / (x_factor * y_factor)

    return out_img.astype(np.uint8)


# reads inputs
img_name = str(input()).rstrip()
save = 1

# reads image from '.' directory
img = imageio.imread(img_name)

img = img.astype(np.float)

out_img = downscale(img, 100, 100)

# verifies if the modified image needs to be saved
if (save):
    imageio.imwrite('output_image.png', out_img.astype(np.uint8))

print("Acabou")
