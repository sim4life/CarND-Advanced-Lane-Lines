import sys, getopt
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from transforms import min_t_pipeline, half_t_pipeline, full_t_pipeline
from grad_color_pipeline import gc_pipeline, gc_pipeline_comb, gc_pipelineV, gc_pipeline_combV, csv_pipeline, gxy_pipeline
from slide_window_hist import slide_window, skip_slide_window, fit_curve
# from conv_window import convolve_window
from lane_curvature import draw_lane_curve, plot_lane_curve, calculate_lane_curve_radius, printHist
from Tracker import Tracker
from video_gen import process_video
from Line import Line

params_out_file = 'output_rsc/wide_dist_pickle.p'
pickle_file     = 'output_rsc/line_fit_pickle.p'
output_dir      = 'output_images/warped'
SKIP_FRAME      = 2
skip_factor     = 0
frame_cnt       = 0
left_fit        = np.array([0,0,0], dtype=np.float64)
right_fit       = np.array([0,0,0], dtype=np.float64)
leftFitLine     = Line()
rightFitLine    = Line()

def gct_bin_pipeline(test_img, params_file, out_dir=output_dir):
    binary_img = gc_pipeline(test_img)#, s_thresh=(170, 245), sx_thresh=(30, 90))
    warped_img, undist_img, perspective_M, perspective_M_inv = min_t_   pipeline(binary_img, single_ch=True, params_file=params_file)
    return warped_img, undist_img, perspective_M, perspective_M_inv

def gct_orig_pipeline(test_img, params_file, out_dir=output_dir):
    warped_img, undist_img, perspective_M, perspective_M_inv = min_t_pipeline(test_img, single_ch=True, params_file=params_file)
    return warped_img, undist_img, perspective_M, perspective_M_inv

def gctc_pipeline(test_img):
    global skip_factor
    params_file = params_out_file
    out_dir = output_dir
    global left_fit
    global right_fit
    global leftFitLine
    global rightFitLine
    global frame_cnt

    warped_img, _, perspective_M, perspective_M_inv = gct_bin_pipeline(test_img, params_file=params_file, out_dir=out_dir)
    _, undistort_img, _, _ = gct_orig_pipeline(test_img, params_file=params_file, out_dir=out_dir)
    if skip_factor <= 0:
        leftx, lefty, rightx, righty, left_fit, right_fit = slide_window(warped_img)
        skip_factor = SKIP_FRAME
        frame_cnt += 1
    else:
        leftx, lefty, rightx, righty, left_fit, right_fit = skip_slide_window(warped_img, left_fit, right_fit)
        skip_factor -= 1
        frame_cnt += 1

    if leftFitLine.isLineDetected(left_fit):
        leftFitLine.addFit(left_fit)
    else:
        leftFitLine.addFit(leftFitLine.avg_lastnfits)
        left_fit = leftFitLine.avg_lastnfits

    if rightFitLine.isLineDetected(right_fit):
        rightFitLine.addFit(right_fit)
    else:
        rightFitLine.addFit(rightFitLine.avg_lastnfits)
        right_fit = rightFitLine.avg_lastnfits

    left_fitx, right_fitx = fit_curve(warped_img, left_fit, right_fit)

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
    # pickle_file =
    op          = 'none'
    test_image  = 'camera_cal/calibration3.jpg'
    op          = 'gct' # perspective transform, gradient masking, color masking
    # skip_factor = 0


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
        leftx, lefty, rightx, righty, left_fit, right_fit = slide_window(warped_img)

        left_fitx, right_fitx = fit_curve(warped_img, left_fit, right_fit)

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

        print('leftFitLine best_fit is:',leftFitLine.best_fit)
        print('rightFitLine best_fit is:',rightFitLine.best_fit)
        vals_pickle = {}
        vals_pickle["recent_lnfits"]   = leftFitLine.recent_nfits
        vals_pickle["lavg_diffs"]      = leftFitLine.avg_diffs
        vals_pickle["lrecent_diffs"]   = leftFitLine.recent_diffs
        vals_pickle["recent_rnfits"]   = rightFitLine.recent_nfits
        vals_pickle["ravg_diffs"]      = rightFitLine.avg_diffs
        vals_pickle["rrecent_diffs"]   = rightFitLine.recent_diffs
        pickle.dump( vals_pickle, open( pickle_file, "wb" ) )

        printHist(leftFitLine.recent_nfits, rightFitLine.recent_nfits)
    elif op == 'gcvxytw':
        # Running gradient_color masking then min transformation pipeline
        warped_img, perspective_M = gcvxyt_pipeline(test_image, params_file=params_file, out_dir=output_dir)
        leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty = slide_window(warped_img)

        skip_slide_window(warped_img, left_fit, right_fit)
    elif op == 'gctwv':
        # Running gradient_color masking then min transformation pipeline
        warped_img, perspective_M = gct_pipelineV(test_image, params_file=params_file, out_dir=output_dir)
        _, _, _, _, left, right, _, _, _ = slide_window(warped_img)
        skip_slide_window(warped_img, left, right)
        # plt.imshow(binary_img, cmap='gray')
        # plt.show()

    elif op == 'gctc':
        # Running gradient_color masking then min transformation pipeline
        warped_img, _, perspective_M, perspective_M_inv = gct_bin_pipeline(test_image, params_file=params_file, out_dir=output_dir)
        _, undistort_img, _, _ = gct_orig_pipeline(test_image, params_file=params_file, out_dir=output_dir)
        # leftx, lefty, rightx, righty, left_fit, right_fit, left_fitx, right_fitx, ploty = slide_window(warped_img)

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

    elif op == 'hist':
        vals_pickle = pickle.load( open( pickle_file, "rb" ) )
        recent_lnfits   = vals_pickle["recent_lnfits"]
        recent_rnfits   = vals_pickle["recent_rnfits"]
        lavg_diffs      = vals_pickle["lavg_diffs"]
        lrecent_diffs   = vals_pickle["lrecent_diffs"]
        ravg_diffs      = vals_pickle["ravg_diffs"]
        rrecent_diffs   = vals_pickle["rrecent_diffs"]

        rr_diffs = np.absolute(rrecent_diffs)
        rr_tmax = np.amax(rr_diffs)
        rr_max = np.amax(rr_diffs, axis=0)
        rr_amax = np.argmax(rr_diffs, axis=0)

        print("len recent_rnfits is:", len(recent_rnfits))
        print("len ravg_diffs is:", len(ravg_diffs))
        print("len rrecent_diffs is:", len(rrecent_diffs))
        print("rr_tempmax is:", rr_tmax)
        print("rr_max is:", rr_max)
        print("rr_amax is:", rr_amax)
        print("diffs_amax is:", rr_diffs[rr_amax])
        # print(rrecent_diffs)
        printHist(recent_lnfits, recent_rnfits)
        printHist(lavg_diffs, ravg_diffs)
        printHist(lrecent_diffs, rrecent_diffs)
    else:
        print("Wrong load option provided: 'all_trans', 'objimg', 'trans', 'gct', 'gctw'")
        sys.exit()


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
