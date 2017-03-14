import numpy as np
import cv2
import matplotlib.pyplot as plt

class Tracker():
    def __init__(self, mywind_width, mywind_height, my_margin, my_ym=1, my_xm=1, mysmooth_factor=15):
        # list to store the prev (left,right) centre set values for smoothing the output
        self.recent_centres = []
        # window pixel width of the centre values, used to count inside centre windows to describe curve values
        self.window_width = mywind_width
        # window pixel height of the centre values, used to count pixels inside centre windows
        # to determine curve values, breaks the image into vertical levels
        self.window_height = mywind_height

        # pixel distance in both directions to slide (left_widnow + right_window) template for searching
        self.margin = my_margin

        self.ym_per_pix = my_ym # meters per pixel in vertical axis
        self.xm_per_pix = my_xm # meters per pixel in horizontal axis
        self.smoothing_factor = mysmooth_factor

    def window_mask(self,width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    # tracking method for finding and storing lane segment positions
    def find_window_centroids(self, binary_warped):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = [] # storing the (left,right) window centre positions per level
        window = np.ones(window_width) # creating a window template to use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum
        # to get the vertical image slice and then np.convolve the vertical image slice with
        # the window template

        # sum quarter bottom of the image to get slice, could use a different ratio
        l_sum = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,:int(binary_warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(binary_warped[int(3*binary_warped.shape[0]/4):,int(binary_warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(binary_warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(binary_warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(binary_warped[int(binary_warped.shape[0]-(level+1)*window_height):int(binary_warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,binary_warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,binary_warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        self.recent_centres.append(window_centroids)
        # returning averaged values of the line centres, helps to keep the markers from jumping around too much
        return np.average(self.recent_centres[-self.smoothing_factor:], axis=0)

    def convolve_window(self, binary_warped):
        window_centroids = self.find_window_centroids(binary_warped)#, self.window_width, self.window_height, self.margin)
        window_width = self.window_width
        window_height = self.window_height

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(binary_warped)
            r_points = np.zeros_like(binary_warped)

            # points used to find the left and right lanes
            leftx = []
            rightx = []

            # Go through each level and draw the windows
            for level in range(0,len(window_centroids)):
                # add centre value found in frame to the list of lane points per left,right
                leftx.append(window_centroids[level][0])
                rightx.append(window_centroids[level][1])

                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
                r_mask = self.window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | (l_mask == 1)] = 255
                r_points[(r_points == 255) | (r_mask == 1)] = 255
            	# l_points[(l_points == 255) | (l_mask == 1)] = 255
            	# r_points[(r_points == 255) | (r_mask == 1)] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channle
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8)

        # Display the final results
        # plt.imshow(output)
        # plt.title('window fitting results')
        # plt.show()
        return output, leftx, rightx

    def draw_lane_lines(self, warped_binary, img, leftx, rightx, M_inv):
        window_width = self.window_width
        window_height = self.window_height
        img_size = (img.shape[1], img.shape[0])
        # fit the lane boundaries to the left, right centre positions found
        yvals = range(0, warped_binary.shape[0])

        res_yvals = np.arange(warped_binary.shape[0]-(window_height/2), 0, -window_height)

        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        middle_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

        road = np.zeros_like(img)
        road_bkg = np.zeros_like(img) # background image
        cv2.fillPoly(road,[left_lane],color=[255,0,0])
        cv2.fillPoly(road,[right_lane],color=[0,0,255])
        cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
        cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

        road_warped = cv2.warpPerspective(road, M_inv, img_size, flags=cv2.INTER_LINEAR)
        road_bkg_warped = cv2.warpPerspective(road_bkg, M_inv, img_size, flags=cv2.INTER_LINEAR)
        img = cv2.addWeighted(img, 1.0, road_bkg_warped, -1.0, 0.0)
        road_warped_wgt = cv2.addWeighted(img, 1.0, road_warped, 1.0, 0.0)

        ym_per_pix = self.ym_per_pix # meters per pixel in y dimension
        xm_per_pix = self.xm_per_pix # meters per pixel in x dimention

        curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
        curvead = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
        # calculate the offset of the car on the road
        camera_centre = (left_fitx[-1] + right_fitx[-1])/2
        centre_diff = (camera_centre-warped_binary.shape[1]/2)*xm_per_pix
        side_pos = 'left'
        if centre_diff <= 0:
            side_pos = 'right'

        # draw the text showing curvature, offset and speed
        cv2.putText(road_warped_wgt,'Radius of Curvature = '+str(round(curvead,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(road_warped_wgt,'Vehicle is '+str(abs(round(centre_diff,3)))+'m '+side_pos+' of centre',(50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        f.tight_layout()
        ax1.imshow(road_warped)
        ax1.set_title('Road warped', fontsize=20)
        ax2.imshow(road_warped_wgt)
        ax2.set_title('Road warped weighted', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

        return road
