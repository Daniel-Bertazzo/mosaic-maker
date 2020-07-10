# SCC0251 Digital Image Processing

## Final Assignment: Reconstructing an image with a Collage Montage

**Authors:**

- Alexandre Norcia Medeiros - nUSP: 10295583
- Daniel Penna Chaves Bertazzo - nUSP: 10349561
- Gabriel Citroni Uliana - nUSP: 9779367

## Main objective

This project aims to create images using other images. It takes two or more images as input (one that will be transformed into a mosaic and the other(s) that will compose the mosaic, referred to as tiles), analyses the former's colors and tries to reconstruct it by changing the tiles' main color. Both images can be of any resolution using the RGB model, and the final image will have the same resolution as the first.

## Examples
**What we were aiming to accomplish:**
![](/images/woman.jpg)

**What we were able to do:**
- Original image:

![](/images/airport.jpg)

- Output image:

![](/output_images/out_airport.jpg)

- Mosaic composition:

![](/output_images/method3pattern.png)

## Images used

The images used to run the program were extracted from the [Pixabay](https://pixabay.com/) website, a royalty free image database.

## Brief discription of how it's done
First, we create the mosaic shape by dividing the image, given a resolution to guide the process. Then, we downscale the images to be used as tiles (compose the mosaic) to the proper size. After this, we iterate over the subimages contained in the mosaic division, obtaining their predominant color. With this color in hand, we alter the predominant color of the tile image to match the given region. Finally, this color-corrected subimage is placed within the final image composition.

In order to accomplish this, we used color analysis to obtain the subimages' predominant color and color balancing to alter the tile images' main color. Moreover, we use an image scaling method (local mean) to downscale the tile images to the correct size. Also, it was used some partition methods to divide the original mosaic into a grid.

## Detailed code description

In order to accomplish a final mosaic, the code was modularized to define the essential operations that will occur in the composition processes. In this way, we define those operations as the mosaic division, tiles downscale, predominant color identification, alter predominant color, and the tile concatenation. The mosaic division operation is the grid construction that defines the surface that each tile represents of the orginal image. The downscale is needed to resize the tiles to the area they need to fill. The predominant color identification picks the most present color in an image, and it occurs alongside the alter color and the tile concatenation functions in order to assemble the mosaic.

First, we read the canvas image (the one that will be turned into a mosaic) and store it in the `canvas_img` variable, along with all of the tile images (stored in the list `tile_imgs`) and the mosaic resolution, which represents the number of tiles in each axis (stored in `x` and `y`) in the final image. Then, we read the variable `division_method`, which is responsible for deciding which method will be used to divide the image into a mosaic. There are four ways to build this grid, in a way that each of them has increasing number of tile unique shapes, in an increasing order (method 1 is the most simple one and method 4 is the most complex, with more distinct tile shapes). In this context, the methods have increasing time complexity, given that for each unique tile shape, it is needed to downscale an image to that resolution, which is a time consuming operation. With this in mind, instead of creating a grid and subsequently downscaling the tile images to each shape simultaneously with the mosaic building process, we were able to save some time by first creating the grid, then associating each tile image to a few random unique shapes and downscaling them only to these pre-selected resolutions, then filling the final mosaic with these already downscaled images. This way, each tile image has only a number of fixed shapes to downscale to, resulting in faster execution, given the fact that the downscale process only happens once per unique shape, instead of "on demand" during the mosaic's assembly. 

The first division method, `division_method=1`, divides the canvas image in a uniform manner(~way/fashion~), in a way that all tiles will have the same width and height, resulting in only one unique division. Note that all division methods will have exactly `(x, y)` divisions, so, in the first method, if the division of the axis size of the canvas image by the `(x,y)` resolution have a remainder greater than zero, there will be some pixels not being included in the division which results in a black border. The black border will be fixed when the input paramenter `fix_black` is equals `1`, this is done by distributing the missing pixels on the initial tiles, and that will generate up to 3 more unique divisions.

- First method division sample:

![](/output_images/method1pattern.png)

For the random mosaic methods (2, 3 and 4), it(~there~) will be used a normal distribution when creating the distincts shapes. To do so, it is necessary the mean and standard deviation to generate the distribution, those parameters are stored in the `mean` and `std` variables, which compose the `rand_parameters=(ux, uy, mean, std)` parameter in the division function. Also, there are the `ux` and `uy` parameters, those parameters are meant to control how many unique shapes will occur on the x and y axis of the divisions, so it can be used to control the code performance.

The second division method, will generate two random vectors called `u` and `v`, these vector have `x` and `y` size respectively and they contain what portion/proportion/rate each division will fill along the axis they represent. All division are get using a combination of a value `u[i]` and `v[j]`. This method generates a random mosaic with few unique division, but the combination of the two vectors creates a impression of some lines and columns outlined/highlighted in the composition. In these methods involving random distributions, the use of floating point operations results in small rounding errors, which ends up causing the presence of a black border in the final mosaic (on the right and bottom sides). This is fixed by getting the size of this border in pixels and increasing the last line and column tiles by this size, thus covering the black border.

- Second method division sample:

![](/output_images/method2pattern.png)

The third method softens the outlines that the second method creates. This is done by generating a new random vector `v` for each row, so it can have unique columns (widths), which causes a misalignment in a column. This method results in a more pleasent mosaic, but it also generates more unique shapes. This method could also be inverted in a way it generates new random `u` vectors, but that results in a unpleasent/unatural/not so great mosaic compared with the diversed `v` vector, so it was not included in the final implementation.

- Third method division sample:

![](/output_images/method3pattern.png)

The fourth method creates a new vector `v` for each row and a new vector `u` for each column, so all elements in the mosaic is more random compared with the other methods. Those shapes causes a misalignment in both axis, making a way more interesting mosaic. This method generates more unique divisions than any other, so it is the most performance demanding. This method also creates black holes in the middle of the mosaic, beacuse the randomness in both axis cause sopreposition and doesnt ensure that all area will be cover. The holes in the mosaic can be seena as weird, but they also cause the impression of a collage of images, with makes a interesting effect when looking the final composition.

- Fourth method division sample:

![](/output_images/method4pattern.png)

After defining the canvas division and performing the downscale of tile images, there is one last step before the final mosaic assembly: matching the tiles to the original image's color in each specific section of the grid. Without this part, the result would just be random small images put together in a senseless collage. In order to do this, we have two functions that will be covered seperately:

`get_predominant_color` - this function is responsible for calculating the predominant color of a given section of the canvas image. It does so by computing the mean of all three color channels separately (R, G and B) within the aforementioned section. Moreover, this function also computes the maximum and minimum values of each channel because they will be used later on. In conclusion, it returns three tuples of three elements each, being one for each color channel: `max_colors`, `min_colors` and `color`.

`alter_color` - this function changes the color of an image to the predominant color computed in the function explained above. It is used to change the color of a given tile to make it match the color of a certain section of the canvas image (stored in `predominant_color`). It has two methods, one uses RGB color representation and the other HSV. In the first method, it multiplies the downscaled tile image's color channels with the `predominant_color` tuple, decreasing the distance between the tile's colors and the original canvas image's colors in that particular section. In the second method, we transform the tile and `predominant_color` into HSV representation. Then, we completely replace the tile's hue and saturation channels with those of `predominant_color`. Afterwards, we transform the tile back to the RGB representation. Regardless of which method is choosed, by the end of the function we normalize the resulting tile back to the original interval of the canvas image's section (that is, we take the max and min values obtained in the `get_predominant_color` function and use them as the new bounds for the tile). This normalization takes the tile's pixels (which are already color corrected to resemble the canvas image) and force their intensity to also match the original image, resulting in a much more accurate tile.

![](/output_images/color_comparison.png)


## Results obtained discussing cases of success and failures

For the random mosaics, we concluded that the normal distribution generates the best mosaic, because its values are often close to the mean. This feature of the normal distribution makes so that all divisions are unique but also not so different in a way it becames an outlier, so the final composition is more pleasant. We also tested the code with exponential and uniform distribution, and both distributions generates wierd divisions that doens't fit well in the overall composition. 

We had some problems changing the tiles to the right color, in both RGB and HSV color methods. When used RGB our tiles tended to get it's colors closer to black, and in the HSV we used to generate really bright images with small color variations. Both methods were resulting in a really poor mosaic, with the colors being far from what we expected. Those problems were all related to the colors intensity and the white and black values, so in order to correct the colors to the right intensity we used a normalization function. We normalize the new colors values to the original color interval that occured in the original canvas image section. Doing so, the RGB values of the altered tile image becomes be fixed in an interval that accurately represent the colors we want to reproduce.

In method 4, as we use random sizes, based in a gaussian, to downscale the tile images, when we insert them to compose the mosaic it results in some black holes in the final image. We tried to do some morphology dilation in the final image to fill those black patches, but the result was unsatisfying because we loose the original shape of each tile. 

We made a loading bar so we can follow the processes of our program in real time, as some of methods can be a bit slow according to the images used. But the loanding bar is not accurate, in some cases it ends before or after its boundary. And specifically on a Jupyter Notebook it's a great formatting problem which makes it have some blank space before it starts print each completition bar.

In order to have multiple images as tiles composing the mosaic, we had to assign which image each division will represent without increasing too much the amount of unique divisions. If we replicate each division to be able to use more than one image, we would end up increasing too much the time required for the program to run. So, what we end up doing was assign only one image to each unique division, in this way we mantain the performance and also have a diversified mosaic. This solution worked really well in the 2 to 4 division methods. The only problem is when there is too little unique division, as in the first method, in those cases the number of divisions limit/bound the number of images.

## Names and descriptions of the roles of every student in the project
This project was developed equally by all the members of the team, using live collaboration tools, like the Live Share extension for Visual Studio Code, which allows for all the members to edit the project code simultaneously and in the same file, and Discord, which allows for live voice communication between participants, as well as screen sharing for better work sharing. Moreover, since we were all working together the whole time, the commits and pushes were made after the meetings were over, and we also tried to balance the number of commits per member.