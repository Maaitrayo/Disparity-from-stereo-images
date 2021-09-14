# ********************************************* IMPORTING MODULES *********************************************
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
# ********************************************* IMPORTING MODULES *********************************************



left_img_file_path = input("Enter the left images folder path:\n") 
right_img_file_path = input("Enter the right images folder path:\n") 
disparity_type = int(input("Enter the type of disparity, 1 for SBGM and 2 for BM:\n "))
disparity_value = int(input("Do you want to print the disparity value(Matrix)?, 1 for YES and 0 for NO:\n "))
image_show = int(input("Do you want to see anyone of the original stereo image?, 1 for YES and 0 for NO:\n "))



# ***************************************************** DATASET HANDLER *****************************************************
# getting the file path for my left image dataset
#path_left = glob.glob("/home/maaitrayo/Autonomous Vehicle/Datasets_short_IMU_GPS_STEREO/2011_09_26_drive_0001_sync/image_00/data/*.png") # LEFT IMAGES
path_left = glob.glob(left_img_file_path+"/*.png") # LEFT IMAGES
# creating a list to store the file path consecutively 
lst_left = []
for file in path_left:
    lst_left.append(file)
# Sorting the images path according to the frame
lst_left.sort()
arr_left = np.array(lst_left)

# getting the file path for my right image dataset
path_right = glob.glob(right_img_file_path+"/*.png") # RIGHT IMAGES
# creating a list to store the file path consecutively
lst_right = []
for file in path_right:
    lst_right.append(file)
# Sorting the images path according to the frame
lst_right.sort()
arr_right = np.array(lst_right)
# ***************************************************** DATASET HANDLER ENDS *****************************************************




# ***************************************************** DISPARITY FOR STEREO IMAGE PAIR HANDLER FUNCTIONS *****************************************************

#                                   ------------------------ Disparity type SBGM ------------------------
def calculate_disparity_SBGM(img_L_g, img_R_g):

    calc_disparity_SBGM = cv2.StereoSGBM_create(minDisparity = 0, 
                                            numDisparities = 96, 
                                            blockSize = 9, 
                                            P1 = 8 * 9 * 9,
                                            P2 = 32 * 9 * 9,
                                            disp12MaxDiff = 1, 
                                            preFilterCap = 63,
                                            uniquenessRatio = 10, 
                                            speckleWindowSize = 100, 
                                            speckleRange = 32)

    img_disparity_SBGM = calc_disparity_SBGM.compute(img_L, img_R)

    if(disparity_value == 1):
        print("--------------------------Disparity value :-------------\n")
        print(img_disparity_SBGM,"\n")
        print("--------------------------Disparity value ends:-------------\n")
    img_disparityA_SBGM = np.divide(img_disparity_SBGM, 255.0)
    cv2.imshow("disparity", img_disparityA_SBGM)
    #plt.imshow(img_disparityA_SBGM)



#                                   ------------------------ Disparity type BM ------------------------
def calculate_disparity_BM(img_L_g, img_R_g):

    calc_disparity_BM = cv2.StereoBM_create(numDisparities = 96,
                                          blockSize = 9)

    img_disparity_BM = calc_disparity_BM.compute(img_L_g, img_R_g)

    if(disparity_value == 1):
        print("--------------------------Disparity value :-------------\n")
        print(img_disparity_BM,"\n")
        print("--------------------------Disparity value ends:-------------\n")
    img_disparityA_BM = np.divide(img_disparity_BM, 255.0)
    cv2.imshow("disparity", img_disparityA_BM)
    #plt.imshow(img_disparityA_BM)

# ***************************************************** DISPARITY FOR STEREO IMAGE PAIR HANDLER FUNCTIONS ENDS *****************************************************



for i in range(len(arr_right)):
    img_L = cv2.imread(arr_left[i])
    img_R = cv2.imread(arr_right[i])

    # Converting to Grayscale/single channel
    img_L_g = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
    img_R_g = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

    if(image_show == 1):
        #Use any one of the image left or right
        cv2.imshow("Image Left", img_L_g)
    #cv2.imshow("Image Right", img_R_g)

    if(disparity_type == 1):
        calculate_disparity_SBGM(img_L_g, img_R_g)

    else:
        calculate_disparity_BM(img_L_g, img_R_g)

    # ------- This portion is just to plot the original image, keep it commented out for smooth workflow of the algorithm -------
    '''plt.subplot(1,2,1)
    plt.title('LEFT IMAGE')
    plt.imshow(img_L_g)

    plt.subplot(1,2,2)
    plt.title('RIGHT IMAGE')
    plt.imshow(img_R_g)'''
    # ------- This portion is just to plot the original image, keep it commented out for smooth workflow of the algorithm -------


    plt.show()
    cv2.waitKey(30)

cv2.destroyAllWindows()