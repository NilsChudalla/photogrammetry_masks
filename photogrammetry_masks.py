import cv2
import numpy as np

class color_mask:
    def __init__(self, name) -> None:
        """
        Constructor of mask.

        name: Name of mask for later usability (str)
        """
        color_mask.name = name




    def define_hsv_limits(self, image_input):
        """
        This function opens a open cv window with sliders. These sliders allow the user to isoalte
        a color they want to mask. Threshold values are written to the color_mask object.

        image_input: path to input image
        """

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        def nothing(x):
            pass

        # Load image
        image = cv2.imread(image_input)

        # Create a window


        # Create trackbars for color change
        # Hue is from 0-179 for Opencv
        cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

        # Set default value for Max HSV trackbars
        cv2.setTrackbarPos('HMax', 'image', 179)
        cv2.setTrackbarPos('SMax', 'image', 255)
        cv2.setTrackbarPos('VMax', 'image', 255)

        # Initialize HSV min/max values
        hMin = sMin = vMin = hMax = sMax = vMax = 0
        phMin = psMin = pvMin = phMax = psMax = pvMax = 0

        while(1):
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            threshhold_dict = {}

            threshhold_dict['hMin'] = hMin
            threshhold_dict['sMin'] = sMin
            threshhold_dict['vMin'] = vMin
            threshhold_dict['hMax'] = hMax
            threshhold_dict['sMax'] = sMax
            threshhold_dict['vMax'] = vMax

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        self.threshholds = threshhold_dict

    def define_mask(self, image_input, kernel=np.ones((5,5))):
        """
        Refine HSV mask with previously set threshold values. Increase amount of dilations with 
        a slider in the opening window.  Iterations either with 5x5 kernel or a custom kerne. 
        Entered values are set for the selected color_mask object. 

        image_input: path to input image (str)
        kernel: dilation kernel, can be e.g. matrix of 3x3, 5x5 or 7x7 ones. 
        """

        threshhold_dict = self.threshholds
        # show result
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)

        def nothing(x):
            pass
        cv2.createTrackbar('Iterations', 'result', 0, 10, nothing)


        image = cv2.imread(image_input)

        # convert image to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # apply mask
        lower_g = np.array([threshhold_dict['hMin'] ,threshhold_dict['sMin'], threshhold_dict['vMin']])
        upper_g = np.array([threshhold_dict['hMax'] ,threshhold_dict['sMax'], threshhold_dict['vMax']])
        mask = cv2.inRange(hsv, lower_g, upper_g)

        # apply dilation to current mask
        kernel = np.ones((5, 5), np.uint8)
        while(1):
            iterations = cv2.getTrackbarPos('Iterations', 'result')
            dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

            # apply mask for result verification
            result = cv2.bitwise_and(image, image, mask=~dilated_mask)

            cv2.imshow('result', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
        cv2.destroyAllWindows()

        self.dilation_iterations = iterations
        self.dilation_kernel = kernel

        return dilated_mask

def apply_combined_masks(masks, image_input, output_dir=False):
    """
    Apply list of masks to image. If output_dir is a path, a mask file with the same of the input image is written to "output_dir".

    masks: list of masks
    image_input: path to input image (str)
    output_dir: False or path to output dir for masks
    """
    


    image = cv2.imread(image_input)

    # convert image to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask_list = []
    
    for imask in masks:
        threshhold_dict = imask.threshholds
        iterations = imask.dilation_iterations
        kernel = imask.dilation_kernel

        # apply mask
        lower_g = np.array([threshhold_dict['hMin'] ,threshhold_dict['sMin'], threshhold_dict['vMin']])
        upper_g = np.array([threshhold_dict['hMax'] ,threshhold_dict['sMax'], threshhold_dict['vMax']])
        mask = cv2.inRange(hsv, lower_g, upper_g)
        dilated_mask = cv2.dilate(mask, kernel, iterations=iterations).astype('bool')

        mask_list.append(dilated_mask)
    
    combi_mask = np.zeros_like(dilated_mask, dtype=bool)

    for imask in mask_list:
        combi_mask += imask

    combi_mask = ~combi_mask.astype(bool)


    if output_dir != False:
        input_dir_names = image_input.split("/")
        file_name = input_dir_names[-1]
        cv2.imwrite(output_dir + '/' + file_name, combi_mask.astype('uint8')*255) 
        print('File %s written' %file_name)
