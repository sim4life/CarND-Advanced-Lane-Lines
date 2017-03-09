import sys, getopt
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from transforms import min_t_pipeline, half_t_pipeline, full_t_pipeline
from grad_color_pipeline import gc_pipeline, gc_pipeline_comb

params_out_file = 'output_rsc/wide_dist_pickle.p'
out_dir         = 'output_images/warped'
def gct_pipeline(test_image, params_file, out_dir=out_dir):
    # image = cv2.imread('gray.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    test_img = cv2.imread(test_image)
    # test_img = mpimg.imread(test_image)
    # binary_img = gc_pipeline(test_img, s_thresh=(170, 245), sx_thresh=(30, 90))
    binary_img = gc_pipeline_comb(test_img)
    plt.imshow(binary_img, cmap='gray')
    plt.show()

    # print("bin image shape is,", binary_img.shape)
    # print("bin image is,", binary_img)
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
        cv2.imwrite(out_dir+'/lane_warped.jpg', warped_img)
    elif op == 'gct':
        # Running gradient_color masking then min transformation pipeline
        warped_img, perspective_M = gct_pipeline(test_image, params_file=params_file, out_dir=out_dir)
    else:
        print("Wrong load option provided: 'all_trans', 'objimg', 'trans', 'gct'")
        sys.exit()


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
