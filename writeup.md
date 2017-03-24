---

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

[image1a]: ./examples/calibration3.jpg "Original"
[image1b]: ./examples/cam3_undist.jpg "Undistorted"
[image2a]: ./test_images/test2.jpg "Road Transformed"
[image2b]: ./examples/lane_undistort_t2.jpg "Road undistorted"
[image2c]: ./examples/lane_binary.jpg "Binary Example"
[image3]: ./examples/lane_curve_t2.png "Warp Example"
[image4]: ./examples/curve_drawn.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 14 through 59 of the file called `transforms.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in the same file, `transforms.py` lines 48 through 59.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image
![alt text][image1a]
Undistorted Image
![alt text][image1b]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2a]
The code undistorting the road image is present in `transforms.py` file, line 101 through 133. In the function, `min_t_pipeline()`, I first load the undistortion matrix and coefficients from a pickle file, `output_rsc/wide_dist_pickle.p`, passed as a parameter, params_file. After that, I pass these parameters to the function, `corners_unwarp_offset()`, to return undistorted image, warped image, perspective transform matrix and perspective transform inverse matrix (for a reverse operation later on). The undistorted road image is as follows:

![alt text][image2b]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 4 through 16 and 137 through 158 in `grad_color_pipeline.py`). I first converted the image to HLS color space and then extracted the S-channel. On this S-channel, I applied the absolute sobel gradient technique on the x-values with the threshold of (20, 100). On this same S-channel, I also applied the threshold of (170, 255) to extract white and yellow lines. Later on, I combined the two binary with an OR operator to return a binary image. Here's an example of my output for this step:

![alt text][image2c]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

As described earlier, the code for my perspective transform exists a function called `corners_unwarp_offset()`, which appears in lines 101 through 127 in the file `transforms.py`.  The `corners_unwarp_offset()` function takes as inputs an image (`img`), as well as perspective matrix (`mtx`) and distortion coefficient (`dist`) and some other paramters.  I chose to find out the trapezoid coordinates based on a hardcoded values of a straight line image and then took offsets and used them to find out the source and destination points in the following manner:

```
	img_size = (gray.shape[1], gray.shape[0])
	mid_width = 0.07 # percent of middle trapeziod width
	height_pct = 0.62 # percent of trapeziod height
	bot_width_r = 0.65 # percent of bottom right trapeziod height
	bot_width_l = 0.614 #percent of bottom left trapeziod width
	bottom_trim = 0.955 # percent from top to bottom to avoid car hood
	offset = img_size[0]*0.25

src = np.float32(
	[[img_size[0]*(0.5-mid_width/2), img_size[1]*height_pct],
	[img_size[0]*(0.5+mid_width/2), img_size[1]*height_pct],
	[img_size[0]*(0.5+bot_width_r/2), img_size[1]*bottom_trim],
	[img_size[0]*(0.5-bot_width_l/2), img_size[1]*bottom_trim]])

dst = np.float32(
	[[offset, 0],
	[img_size[0]-offset, 0],
	[img_size[0]-offset, img_size[1]],
	[offset, img_size[1]]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 449      | 320, 0        | 
| 684, 449      | 960, 0      |
| 1056, 687     | 960, 720      |
| 247, 687      | 320, 720        |


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Afterwards, I applied the slide window technique to fit my lane lines with a 2nd order polynomial. The code is in file `slide_window_hist.py` from lines 5 to 116. I'm using using the function `slide_window()` to find the lane line 2nd order polynomial with 9 sliding windows for one frame. For the next frame, I am skipping fitting the lane lines by using the left line fit and right line fit of the previous frame and applying it to fit the 2nd order polynomial in the current frame in the function `skip_slide_window()`. The results can be seen in the following image:

![alt text][image3]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 41 through 64 in my function `calculate_lane_curve_radius()` and in lines 66 through 101 in my function `draw_lane_curve()` in the file `lane_curvature.py`. I used the earlier fitted left lane and right lane 2nd order polynomial coefficients to calculate the left line and right line curvature in the real world coordinates using the constants:

```
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

```

After that, in the function `draw_lane_curve()`, I used the left line and the right line curvature to draw it on the warped image Then, I used the undistorted image, binary warped image and the inverse perspective matrix to unwarp the image. Finally, I calculated the real world line curvature and distance from centre based on the earlier calculated real world left line curvature only.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 9 through 17 in my code in `video_gen.py` in the function `process_video()`.  The main file to run all the different function correctly is `lane-tracker.py` lines 141 through 175. Here is an example of my result on a test image:

![alt text][image4]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_proc.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

