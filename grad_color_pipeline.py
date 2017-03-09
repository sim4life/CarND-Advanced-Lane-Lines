import numpy as np
import cv2

def gc_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    hls_s_channel = hls[:,:,2]

    # plt.imshow(hls_s_channel, cmap='gray')
    # plt.show()
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    # hsv_v_channel = hsv[:,:,2]

    sxbinary = abs_sobel_thresh(hls_s_channel, orient='x', sobel_kernel=3, thresh=sx_thresh)
    s_binary = channel_thresh(hls_s_channel, s_thresh)
    # sxbinary = abs_sobel_thresh(hsv_v_channel, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # s_binary = channel_thresh(hsv_v_channel, s_thresh)

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def gc_pipeline_comb(img, s_thresh=(170, 255), xy_th=(20, 100), mag_th=(20, 100), dir_th=(0.7, 1.3)):
    img = np.copy(img)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    hls_s_channel = hls[:,:,2]

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
    # hsv_s_channel = hsv[:,:,1]

    sxbinary = combined_thresh(hls_s_channel, xy_thresh=xy_th, m_thresh=mag_th, d_thresh=dir_th)
    s_binary = channel_thresh(hls_s_channel, s_thresh)
    # sxbinary = combined_thresh(hsv_s_channel, xy_thresh=xy_th, m_thresh=mag_th, d_thresh=dir_th)
    # s_binary = channel_thresh(hsv_s_channel, s_thresh)

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

def mag_thresh(channel, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # gray = cv2.cvtColor(channel, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(channel, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >= dir_thresh[0]) & (dir_grad <= dir_thresh[1])] = 1
    return dir_binary

def combined_thresh(channel, xy_thresh=(20, 100), m_thresh=(30, 100), d_thresh=(0.7, 1.3)):
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    # Run the functions
    gradx = abs_sobel_thresh(channel, orient='x', sobel_kernel=ksize, thresh=xy_thresh)
    grady = abs_sobel_thresh(channel, orient='y', sobel_kernel=ksize, thresh=xy_thresh)
    mag_binary = mag_thresh(channel, sobel_kernel=9, mag_thresh=m_thresh)
    dir_binary = dir_threshold(channel, sobel_kernel=15, dir_thresh=d_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined

def channel_thresh(channel, s_thresh=(170, 255)):
    # Threshold color channel
    s_binary = np.zeros_like(channel)
    s_binary[(channel >= s_thresh[0]) & (channel <= s_thresh[1])] = 1

    return s_binary

def abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply threshold
    if orient is 'x':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient is 'y':
        abs_sobel = np.absolute(cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    else:
        print("Wrong orienatation provided in abs_sobel_thresh()")
        return

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary
