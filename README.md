# SCC0251 Digital Image Processing

## Final Assignment: Reconstructing an image with a Collage Montage

**Authors:**

- Alexandre Norcia Medeiros - nUSP: 10295583
- Daniel Penna Chaves Bertazzo - nUSP: 10349561
- Gabriel Citroni Uliana - nUSP: 9779367

## Main objective

This project aims to create images using other images. It takes two images as input (the one that will be transformed into a mosaic and the one that will compose the mosaic), analyses the former's colors and tries to reconstruct it by changing the latter's main color. Both images can be of any resolution using the RGB model, and the final image will have the same resolution as the first.

## Examples
**What we are aiming to accomplish:**
![](/images/woman.jpg)

**What we were able to do so far:**
- Image to be turned into mosaic:
![](/images/lapis2.jpg)

- Image to compose the mosaic:
![](/images/verde_p.png)

- Output image:
![](/output_images/output_test_linear.png)


## Possible features for the future

- Enhance the color balancing applied on the subimages;
- Creates a mosaic with more varied shapes, with rectangles of different dimensions;
- Creates a mosaic composed by more than one image;
- Different method for defining the subimage's predominant color (using histograms);
- Different method for changing the image's predominant color (using HSI/HSV instead of RGB);
- Maybe apply a filter to enhance the final image.

## Images used

The images being used to run the program are being extracted from the [Pixabay](https://pixabay.com/) website, a royalty free image database.

## How it's done
First, we create the mosaic shape by dividing the image, given a resolution to guide the process. Then, we downscale the images to be used as tiles (compose the mosaic) to the proper size. After this, we iterate over the subimages contained in the mosaic division, obtaining their predominant color. With this color in hand, we alter the predominant color of the tile image to match the given region. Finally, this color-corrected subimage is placed within the final image composition.

In order to accomplish this, we used color analysis to obtain the subimages' predominant color and color balancing to alter the tile images' main color. Moreover, we use an image scaling method (local mean) to downscale the tile images to the correct size.