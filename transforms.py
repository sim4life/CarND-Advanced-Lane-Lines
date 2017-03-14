import cv2
import glob
import pickle
import numpy as np

nx, ny = 9, 6

img_in_dir      = 'camera_cal'
chess_out_dir   = 'output_images/draw_chess'
calib_out_dir   = 'output_images/calib_cam'
warp_out_dir    = 'output_images/warped'
# params_file    = 'wide_dist_pickle.p'

def get_obj_img_points(nx=9, ny=6, in_dir='camera_cal', out_dir='draw_chess'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    # Make a list of calibration images
    images = glob.glob(in_dir+'/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            write_name = out_dir + '/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()
    return objpoints, imgpoints

def calibrate_undistort(objpoints, imgpoints, test_image, out_dir='calib_cam'):
    # Test undistortion on an image
    # img = cv2.imread(test_image)
    print("test_image type,", type(test_image))
    img_size = (test_image.shape[1], test_image.shape[0])

    # Doing camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = cv2.undistort(test_image, mtx, dist, None, mtx)
    cv2.imwrite(out_dir+'/lane_undist.jpg', dst)
    return mtx, dist

def save_params(objpoints, imgpoints, mtx, dist, out_file='wide_dist_pickle.p'):
    # Saving the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( out_file, "wb" ) )

def load_params(in_file='wide_dist_pickle.p'):
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( in_file, "rb" ) )
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    mtx       = dist_pickle["mtx"]
    dist      = dist_pickle["dist"]
    return objpoints, imgpoints, mtx, dist

def corners_unwarp(img, nx, ny, mtx, dist, single_ch=False, out_dir='warped'):

    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(out_dir+'/lane_undistort.jpg', undist_img)

    gray = img
    if single_ch == False:
        gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)

    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    # For source points I'm grabbing the trapeziod coordinates of two lane lines
    # src = np.float32([[615, 437], [663, 437], [1057, 688], [247, 688]])
    src = np.float32([[595, 449], [684, 449], [1057, 688], [247, 688]])
    # For destination points, I'm choosing some points based on undistorted image

    # dst = np.float32([[270, 0], [1040, 0], [1040, img_size[1]], [270, img_size[1]]])
    dst = np.float32([[247, 0], [1057, 0], [1057, img_size[1]], [247, img_size[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, M_inv

def corners_unwarp_offset(img, nx, ny, mtx, dist, single_ch=False, out_dir='warped'):

    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    gray = img
    if single_ch == False:
        gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)

    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])
    ## bot_width = 0.633 # 0.76 # percent of bottom trapeziod height
    mid_width = 0.07 # 0.08 # percent of middle trapeziod width
    height_pct = 0.62 # percent of trapeziod height
    bot_width_r = 0.65 # 0.76 # percent of bottom right trapeziod height
    bot_width_l = 0.614 #percent of bottom left trapeziod width
    bottom_trim = 0.955 #0.935 # percent from top to bottom to avoid car hood
    # bot_width_r = 0.628 # original: percent of bottom right trapeziod width
    # bot_width_l = 0.58 # original: percent of bottom left trapeziod width
    # bottom_trim = 0.936 # original: percent from top to bottom to avoid car hood

    # For source points I'm grabbing the trapeziod coordinates of two lane lines
    # src = np.float32([[595, 449], [684, 449], [1057, 688], [247, 688]]) # undistored image
    # src = np.float32([[613, 437], [664, 437], [1042, 674], [268, 674]]) # original image
    src = np.float32([[gray.shape[1]*(0.5-mid_width/2), gray.shape[0]*height_pct],[gray.shape[1]*(0.5+mid_width/2), gray.shape[0]*height_pct], [gray.shape[1]*(0.5+bot_width_r/2), gray.shape[0]*bottom_trim], [gray.shape[1]*(0.5-bot_width_l/2), gray.shape[0]*bottom_trim]])
    offset = img_size[0]*0.25

    # For destination points, I'm choosing some points based on undistorted image
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    # dst = np.float32([[247, 0], [1057, 0], [1057, img_size[1]], [247, img_size[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, undist_img, M, M_inv

def min_t_pipeline(test_img, params_file, single_ch=False):
    # Reading in the saved objpoints, imgpoints, mtx, dist
    objpoints, imgpoints, mtx, dist = load_params(in_file=params_file)
    warped_img, undist_img, perspective_M, perspective_M_inv = corners_unwarp_offset(test_img, nx, ny, mtx, dist, single_ch, out_dir=warp_out_dir)
    return warped_img, undist_img, perspective_M, perspective_M_inv

def half_t_pipeline(image_filename, params_file):
    test_img = cv2.imread(image_filename)
    print("test_image type,", type(test_image))
    # Reading in the saved objpoints, imgpoints:: mtx and dist are dummy
    objpoints, imgpoints, mtx, dist = load_params(in_file=params_file)
    mtx, dist = calibrate_undistort(objpoints, imgpoints, test_image=test_img, out_dir=calib_out_dir)
    save_params(objpoints=objpoints, imgpoints=imgpoints, mtx=mtx, dist=dist, out_file=params_file)
    warped_img, perspective_M = corners_unwarp(test_img, nx, ny, mtx, dist, out_dir=warp_out_dir)
    return warped_img, perspective_M

def full_t_pipeline(image_filename, params_file):
    objpoints, imgpoints = get_obj_img_points(nx=nx, ny=ny, in_dir=img_in_dir, out_dir=chess_out_dir)
    test_img = cv2.imread(image_filename)
    mtx, dist = calibrate_undistort(objpoints, imgpoints, test_image=test_img, out_dir=calib_out_dir)
    # Writing out the params objpoints, imgpoints, mtx and dist
    save_params(objpoints=objpoints, imgpoints=imgpoints, mtx=mtx, dist=dist, out_file=params_file)
    warped_img, perspective_M = corners_unwarp(test_img, nx, ny, mtx, dist, out_dir=warp_out_dir)
    return warped_img, perspective_M
