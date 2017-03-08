import cv2
import glob
import pickle
import sys, getopt
import numpy as np

nx, ny = 9, 6

img_in_dir      = 'camera_cal'
chess_out_dir   = 'output_images/draw_chess'
calib_out_dir   = 'output_images/calib_cam'
warp_out_dir    = 'output_images/warped'
params_out_file = 'output_rsc/wide_dist_pickle.p'
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
    img_size = (test_image.shape[1], test_image.shape[0])

    # Doing camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = cv2.undistort(test_image, mtx, dist, None, mtx)
    cv2.imwrite(out_dir+'/test_undist.jpg', dst)
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

def corners_unwarp(img, nx, ny, mtx, dist, out_dir='warped'):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found:
            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view

    # img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    gray = cv2.cvtColor(undist_img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners - just for fun!!!
        cv2.drawChessboardCorners(undist_img, (nx, ny), corners, ret)
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)
        cv2.imwrite(out_dir+'/test_warped.jpg', warped)
    return warped, M

def min_pipeline(image_filename, params_file):
    test_img = cv2.imread(image_filename)
    # Reading in the saved objpoints, imgpoints, mtx, dist
    objpoints, imgpoints, mtx, dist = load_params(in_file=params_file)
    top_down, perspective_M = corners_unwarp(test_img, nx, ny, mtx, dist, out_dir=warp_out_dir)
    return top_down, perspective_M

def half_pipeline(image_filename, params_file):
    test_img = cv2.imread(image_filename)
    # Reading in the saved objpoints, imgpoints:: mtx and dist are dummy
    objpoints, imgpoints, mtx, dist = load_params(in_file=params_file)
    mtx, dist = calibrate_undistort(objpoints, imgpoints, test_image=test_img, out_dir=calib_out_dir)
    save_params(objpoints=objpoints, imgpoints=imgpoints, mtx=mtx, dist=dist, out_file=params_file)
    top_down, perspective_M = corners_unwarp(test_img, nx, ny, mtx, dist, out_dir=warp_out_dir)
    return top_down, perspective_M

def full_pipeline(image_filename, params_file):
    objpoints, imgpoints = get_obj_img_points(nx=nx, ny=ny, in_dir=img_in_dir, out_dir=chess_out_dir)
    test_img = cv2.imread(image_filename)
    mtx, dist = calibrate_undistort(objpoints, imgpoints, test_image=test_img, out_dir=calib_out_dir)
    # Writing out the params objpoints, imgpoints, mtx and dist
    save_params(objpoints=objpoints, imgpoints=imgpoints, mtx=mtx, dist=dist, out_file=params_file)
    top_down, perspective_M = corners_unwarp(test_img, nx, ny, mtx, dist, out_dir=warp_out_dir)
    return top_down, perspective_M

def main(argv):
    params_file = params_out_file
    load_opt   = 'none'
    test_image = 'calibration3.jpg'

    try:
        opts, args = getopt.getopt(argv,"hp:l:t:",["pfile=","load=","testimg="])
        print("opts are:",opts)
        print("args are:",args)
    except getopt.GetoptError:
        print ('transforms.py -p <paramsfile> -l <load_option> -t <test_image')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('transforms.py -p <paramsfile> -l <load_option> -t <test_image>')
            sys.exit()
        elif opt in ("-p", "--pfile"):
            params_file = arg
        elif opt in ("-l", "--load"):
            load_opt = arg
        elif opt in ("-t", "--testimg"):
            # print('arg is:', arg)
            test_image = arg

    print ('Params file is "', params_file)
    print ('Load is "', load_opt)
    print ('Test image is "', test_image)

    if load_opt == 'all':
        # Running minimum pipeline
        top_down, perspective_M = min_pipeline(img_in_dir+'/'+test_image, params_file=params_file)
    elif load_opt == 'objimg':
        # Running half pipeline
        top_down, perspective_M = half_pipeline(img_in_dir+'/'+test_image, params_file=params_file)
    elif load_opt == 'none':
        # Running full pipeline
        ftop_down, perspective_M = full_pipeline(img_in_dir+'/'+test_image, params_file=params_file)
    else:
        print("Wrong load option provided: 'all', 'objimg', 'none'")
        sys.exit()


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
