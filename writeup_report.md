## Writeup Report

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboard]: ./images/chessboard.png "Chessboard"
[undistorted]: ./images/undistorted.png "Undistorted"
[binary_mask]: ./images/binary_mask.png "Binary mask"
[transformation]: ./images/transformation.png "Transformation"
[challenge_3]: ./test_images/challenge_3.jpg "Challenge 3"
[polyfit]: ./images/polyfit.png "Polyfit"
[pipeline]: ./images/pipeline.png "Pipeline"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the [IPython notebook](advanced_lane_finding.ipynb), `calibrate_camera` function.  

I start by preparing `points`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_points` is just a replicated array of coordinates, and `points ` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points ` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![chessboard]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first step of image processing is remove camera distortion by applying transformations calculated during camera calibration. You can see the results on one of the test images below.

![undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (see `to_binary` function in [IPython notebook](advanced_lane_finding.ipynb)). 
I based the code on the examples from the learning materials (combining Sobel X on grayscale image with thresholding Saturation channel from HLS image representation) and made the following changes:

- Instead of grayscale image I used Red color channel as a source for Sobel X function (as lane lines were most prominently visible on the Red color channel)
- Tuned thresholding limits a little bit
- Added Vibrance channel (from HSV representation) to the logic: only pixels with Vibrance higher than `110` are included
- For Vibrance channel thresholding I've removed border pixels (which have Vibrance value higher than thresholding limit, but also have neighbours with Vibrance value below the threshold). To do this I've used blur on vibrance channel and then applying thresholding again.

```python
# Threshold vibrance channel
v_thresh_min = 110
v_thresh_max = 255
v_channel[v_channel < v_thresh_min] = 0
v_channel = cv2.blur(v_channel, (5,5))
v_binary = np.zeros_like(v_channel)
v_binary[(v_channel >= v_thresh_min) & (v_channel <= v_thresh_max)] = 1
```

You can see the illustration of this logic below. Sobel X thresholding is represented by the green color dots, Saturation thresholding -- by the blue dots and Vibrance thresholding are red. And the final binary image contains dots which are both red and green or red and blue (which are yellow and magenta on the image below).

![binary_mask]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Image perspective is corrected using OpenCV `warpPerspective` method. To construct transformation matrix I've used source and destination rects from lecture materials, but after applying them to the images with straighs lines (from test image set) I've tuned the rects up a little so the lines actually appear straight after transformation. The final code is in `transform_perspective` function (it also supports optional `inverse` parameter to transfotm perspective back).

Here are my values for the transformation source/destination coordinates:

```python
src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 20), img_size[1]],
        [(img_size[0] * 5 / 6) + 20, img_size[1]],
        [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])

dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 193, 720      | 320, 720      |
| 1087, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![transformation]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I've started with the algorithm from the learning materials for identifying lane lines in the first and consecutive video images. Here is a short overview of the original algorithm:

1. Make a histogram of the lower part of the binary mask image (sum over y axis) and identify two peaks on the left and on the right
2. Use those peaks as a center X coordinate of non-zero pixels identification window
3. Slide the window from the bottom of the image to the top. Use average X coordinate of non-zero pixels of the previous window (if there are enough of those, we have a constant for this in the code) as a new window center coordinate. Save non-zero pixels on each step
4. Use all non-zero pixels from all the windows to fit a second degree polynomial. Important detail: our polynomial is a function of Y and not of X (as it usually is) because we need to support vertical lines and polynomial function of X coordinate for vertical line does not exist.

...and for the rest of the frames we skip steps 1 through 3 and instead use polynomial from the previous frame as a center of the window where we take our non-zero pixels from.

After experimenting with different images I've made the following modifications to the algorithm:

- After half of the image is processed we try to fit polynomial (through the points we already have) after each window and then return the average of those polynomials
- If there are not enough non-zero pixels for identifying the new center of the next window (and we've already calculated half-way polynomial), then instead of keeping the same center in the next window we evaluate X coordinate of our latest polynomial estimation and use resulting X value instead

The main reason for those changes is that while binary image almost always has lane lines correctly identified on the bottom of the image, on the top of the image it frequently loses the lane lines and original algorithm picked up random pixels (for example, from road defects) and tried to fit polynomial through those random points which resulted in very strange shapes sometimes.

Here is an illustration of how the updated algorithm works on particularly complex image taken from the challenge video:

![challenge_3]

![polyfit]

If not for my changes the right lane would have included pixels from the middle of the road, but moving vindow according to the new rules skips those pixels.

I've also made improvements to the algorithm finding the lines in consequent video frames. Here is the list:

- Instead of always using all points aroung previous curves we now take 4 windows: bottom, top, middle and whole image, calculate 4 polynomials and keep the one closest to the previous polynomial (we use first coefficient when looking for the closest polymer because it affects the shape of the curve the most)
- After new fit is found we use exponential smoothing (`new_value = new_value * a + old_value * (1 - a)`) for the return value. This reduces lane lines shaking because of the rapid camera movements and lightning conditions changes

The final code is located in `get_lines` and `get_next_lines` functions.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done in `get_curvature_and_offset`. For curvature we use exactly the same method as described in lecture materials (I applied the formula given to the polynomial coefficients).

As for the vehicle offset from the center of the road, I evaluated both line positions on the bottom of the image, then took average of them (thus calculating the center of the lane) and then subtracted the actual center point of the image. The resulting value was multiplied by the number of meters per pixel on X axis. Here is the final code for calculating the offset:

```python
# Calculating offset
y_eval = image_height - 1

left_bottom = np.polyval(left_fit, y_eval)
right_bottom = np.polyval(right_fit, y_eval)
lane_center = (right_bottom + left_bottom) / 2
image_center = image_width / 2

offset = (image_center - lane_center) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I've combined all the steps above in `process_image` function. It does the following:

- Uses camera calibration (triggered by helper function `reset_pipeline`) to undistort the image
- Convert undistorted image to the binary mask
- Fix perspective for the resulting image
- Find polynomials for the left and right lane lines
- Draw the lane using polynomials, transform back the perspective and place it on top of the original image
- Find curvature and vehicle offset and print it on top of the image

You can see the result on one of the test images below:

![pipeline]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Resulting pipeline works well on the [project video](test_videos_output/project_video.mp4) and reasonably well on the [challenge video](test_videos_output/challenge_video.mp4).

[![Project Video](http://img.youtube.com/vi/3bJAKXk71Zc/0.jpg)](http://www.youtube.com/watch?v=3bJAKXk71Zc)

[![Challenge Video](http://img.youtube.com/vi/UAx43GP54yw/0.jpg)](http://www.youtube.com/watch?v=UAx43GP54yw)

Applying my pipeline in its current state to the harder challenge video did not look good, so I'm not including it here.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I've already described the changes I've done to the algorithm to make it more robust. So here I'll list the things I found to be difficult for this pipeline to process:

- The same problems I listed with the first project in this nanodegree (basic lane finding): thresholding values used for line identification will depend on lighting and atmospheric conditions when the images are taken. For images taken during the night or when raining our pipeline will likely fail
- More lines on the image (like road defect on the middle of the road in the challenge video or other random lines on the road) can confuse the pipeline. Actually this was one of the problems my changes to the line finding algorithm solved for the challenge video, but it still fails on the harder challenge video
- Pipeline expects that car is approximately on the middle of the lane. If car is crossing the lane we will not be able to find lane lines (algorithm looks for exactly two lines, on on the left and one on the right)
- There probably will be issues on videos taken on uneven road. If elevation is the same throughout the video we can correct the perspective using constant frame coordinates, but for ascending/descending road we would need some way to adjust for the road angle

And to make this pipeline better we can:

- Test and tune binary masking on more test data: different roads, different conditions, etc
- We can try to update the algorithm to identify different number of lane lines (less and more than 2). To do this we will need to modify the initial step (where we identify sliding windows for lane lines). And finding lines in consecutive frames can essentiall stay the same
- Try to take more videos of the same road with more cameras and use the lines from the best-fit camera
- And one of the cameras could be infrared for the low light conditions (as described [here](http://www.temjournal.com/content/41/06/temjournal4106.pdf))

Overall the pipeline designed in this project is much better than the one I designed for the first project of this nanodegree, but I still feel that it has a lot of places where it can be improved.