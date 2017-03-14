import sys, getopt
import cv2
import numpy as np
import matplotlib.pyplot as plt

from transforms import min_t_pipeline, half_t_pipeline, full_t_pipeline
from grad_color_pipeline import gc_pipeline, gc_pipeline_comb, gc_pipelineV, gc_pipeline_combV, csv_pipeline, gxy_pipeline
from slide_window_hist import slide_window, skip_slide_window
# from conv_window import convolve_window
from lane_curvature import draw_lane_curve, plot_lane_curve, calculate_lane_curve_radius
from Tracker import Tracker
from video_gen import process_video

params_out_file = 'output_rsc/wide_dist_pickle.p'
output_dir         = 'output_images/warped'
def gct_bin_pipeline(test_img, params_file, out_dir=output_dir):
    binary_img = gc_pipeline(test_img)#, s_thresh=(170, 245), sx_thresh=(30, 90))
    warped_img, undist_img, perspective_M, perspective_M_inv = min_t_pipeline(binary_img, single_ch=True, params_file=params_file)
    return warped_img, undist_img, perspective_M, perspective_M_inv

def gct_orig_pipeline(test_img, params_file, out_dir=output_dir):
    warped_img, undist_img, perspective_M, perspective_M_inv = min_t_pipeline(test_img, single_ch=True, params_file=params_file)
    return warped_img, undist_img, perspective_M, perspective_M_inv

def gctc_pipeline(test_img):
    params_file = params_out_file
    out_dir = output_dir
    warped_img, _, perspective_M, perspective_M_inv = gct_bin_pipeline(test_img, params_file=params_file, out_dir=out_dir)
    _, undistort_img, _, _ = gct_orig_pipeline(test_img, params_file=params_file, out_dir=out_dir)
    leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, ploty = slide_window(warped_img)

    # skip_slide_window(warped_img, left_fit, right_fit)
    # convolve_window((warped_img*255))
    # plot_lane_curve(leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, ploty)
    # plot_lane_curve2()

    left_curverad, right_curverad = calculate_lane_curve_radius(warped_img, left_fit, right_fit)
    result = draw_lane_curve(undistort_img, warped_img, perspective_M_inv, left_fitx, right_fitx, left_curverad)
    return result

def gcvxyt_pipeline(test_image, params_file, out_dir=output_dir):
    test_img = cv2.imread(test_image)
    binary_imgcv = csv_pipeline(test_img, s_thresh=(100, 255), v_thresh=(50, 255))
    binary_imgxy = gxy_pipeline(test_img, x_thresh=(12, 255), y_thresh=(25, 255))#, s_thresh=(170, 245), sx_thresh=(30, 90))
    binary_img = np.zeros_like(binary_imgcv)
    binary_img[(binary_imgcv == 1) | (binary_imgxy == 1)] = 1

    cv2.imwrite(out_dir+'/lane_binary.jpg', (binary_img*255))
    warped_img, perspective_M, perspective_M_inv = min_t_pipeline(binary_img, single_ch=True, params_file=params_file)
    cv2.imwrite(out_dir+'/lane_gc_warped.jpg', (warped_img*255))
    return warped_img, perspective_M

def gct_pipelineV(test_image, params_file, out_dir=output_dir):
    test_img = cv2.imread(test_image)
    binary_img = gc_pipelineV(test_img)#, s_thresh=(170, 245), sx_thresh=(30, 90))
    binary_img = gc_pipeline_combV(test_img)#, s_thresh=(170, 245), xy_th=(20, 100), mag_th=(40, 100), dir_th=(0.7, 1.3))

    cv2.imwrite(out_dir+'/lane_binary.jpg', (binary_img*255))
    warped_img, perspective_M = min_t_pipeline(binary_img, single_ch=True, params_file=params_file)
    cv2.imwrite(out_dir+'/lane_gc_warped.jpg', (warped_img*255))

    return warped_img, perspective_M

def main(argv):
    params_file = params_out_file
    op          = 'none'
    test_image  = 'camera_cal/calibration3.jpg'
    op          = 'gct' # perspective transform, gradient masking, color masking


    try:
        opts, args = getopt.getopt(argv,"hp:o:t:",["pfile=","op=","testimg="])
    except getopt.GetoptError:
        print ('lane-tracker.py -p <paramsfile> -o <operation> -t <test_image>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('lane-tracker.py -p <paramsfile> -o <operation> -t <test_image>')
            sys.exit()
        elif opt in ("-p", "--pfile"):
            params_file = arg
        elif opt in ("-o", "--op"):
            op = arg
        elif opt in ("-t", "--testimg"):
            test_image = arg

    print ('Params file is "', params_file)
    print ('Test image is "', test_image)
    print ('Operation is "', op)

    if op == 'all_trans':
        # Running full transformation pipeline
        warped_img, perspective_M = full_t_pipeline(test_image, params_file=params_file)
    elif op == 'objimg':
        # Running half transformation pipeline
        warped_img, perspective_M = half_t_pipeline(test_image, params_file=params_file)
    elif op == 'trans':
        test_img = cv2.imread(test_image)
        # Running full transformation pipeline
        warped_img, perspective_M = min_t_pipeline(test_img, params_file=params_file)
        cv2.imwrite(output_dir+'/lane_warped.jpg', warped_img)
    elif op == 'gct':
        # Running gradient_color masking then min transformation pipeline
        warped_img, perspective_M = gct_pipeline(test_image, params_file=params_file, out_dir=output_dir)
    elif op == 'gctw':
        test_img = cv2.imread(test_image)
        # Running gradient_color masking then min transformation pipeline
        warped_img, _, perspective_M, perspective_M_inv = gct_bin_pipeline(test_img, params_file=params_file, out_dir=output_dir)
        _, undistort_img, _, _ = gct_orig_pipeline(test_img, params_file=params_file, out_dir=output_dir)
        leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, ploty = slide_window(warped_img)

        # skip_slide_window(warped_img, left_fit, right_fit)
        # convolve_window((warped_img*255))
        # plot_lane_curve(leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, ploty)
        # plot_lane_curve2()

        left_curverad, right_curverad = calculate_lane_curve_radius(warped_img, left_fit, right_fit)
        result = draw_lane_curve(undistort_img, warped_img, perspective_M_inv, left_fitx, right_fitx, left_curverad)
        plt.imshow(result)
        plt.title('window fitting results')
        plt.show()

    elif op == 'gctwvid':
        input_video_file = test_image
        output_video_file = input_video_file.split(".")[0] + '_proc.' + input_video_file.split(".")[1]
        print("output_video_file =", output_video_file)
        # test_img = cv2.imread(test_image)
    
        process_video(process_image=gctc_pipeline, params_file=params_file, in_vid=input_video_file, out_vid=output_video_file)

    elif op == 'gcvxytw':
        # Running gradient_color masking then min transformation pipeline
        warped_img, perspective_M = gcvxyt_pipeline(test_image, params_file=params_file, out_dir=output_dir)
        leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, ploty = slide_window(warped_img)

        skip_slide_window(warped_img, left_fit, right_fit)
    elif op == 'gctwv':
        # Running gradient_color masking then min transformation pipeline
        warped_img, perspective_M = gct_pipelineV(test_image, params_file=params_file, out_dir=output_dir)
        _, _, left, right, _, _, _ = slide_window(warped_img)
        skip_slide_window(warped_img, left, right)
        # plt.imshow(binary_img, cmap='gray')
        # plt.show()

    elif op == 'gctc':
        # Running gradient_color masking then min transformation pipeline
        warped_img, _, perspective_M, perspective_M_inv = gct_bin_pipeline(test_image, params_file=params_file, out_dir=output_dir)
        _, undistort_img, _, _ = gct_orig_pipeline(test_image, params_file=params_file, out_dir=output_dir)
        # leftx, rightx, left_fit, right_fit, left_fitx, right_fitx, ploty = slide_window(warped_img)

        # skip_slide_window(warped_img, left_fit, right_fit)
        # convolve_window((warped_img*255))
        # window settings
        window_width = 50
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        margin = 100 # How much to slide left and right for searching

        curve_centres = Tracker(mywind_width=window_width, mywind_height=window_height, my_margin=margin, my_ym=10/720, my_xm=4/384, mysmooth_factor=15)
        # window_centroids = curve_centres.find_window_centroids(warped_img)
        convolve_img, leftx, rightx = curve_centres.convolve_window(warped_img)
        # Display the final results
        # plt.imshow(convolve_img)
        # plt.title('window fitting results')
        # plt.show()
        test_img = cv2.imread(test_image)

        curve_centres.draw_lane_lines(warped_img, test_img, leftx, rightx, perspective_M_inv)

    else:
        print("Wrong load option provided: 'all_trans', 'objimg', 'trans', 'gct', 'gctw'")
        sys.exit()


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
