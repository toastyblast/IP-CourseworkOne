# -*- coding: utf-8 -*-
"""
Wrap the OpenPose library with Python.
To install run `make install` and library will be stored in /usr/local/python

TODO: NOTE FOR TEACHERS OR OTHERS USING THIS CODE:
    Scroll down to the bottom of this file to choose if you want to run the
     code for Task 2 or Task 4, as both of these programs' code are included in this single file!
"""
import math

import numpy as np
import ctypes as ct
import cv2
import os
from sys import platform
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    os.environ['PATH'] = dir_path + "/../../bin;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Debug;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Release;" + os.environ['PATH']

class OpenPose(object):
    """
    Ctypes linkage
    """
    if platform == "linux" or platform == "linux2":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    elif platform == "darwin":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.dylib')
    elif platform == "win32":
        try:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Release/_openpose.dll')
        except OSError as e:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Debug/_openpose.dll')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8), ct.c_bool]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.poseFromHeatmap.restype = None

    def encode(self, string):
        return ct.c_char_p(string.encode('utf-8'))

    def __init__(self, params):
        """
        OpenPose Constructor: Prepares OpenPose object

        Parameters
        ----------
        params : dict of required parameters. refer to openpose example for more details

        Returns
        -------
        outs: OpenPose object
        """
        self.op = self._libop.newOP(params["logging_level"],
		                            self.encode(params["output_resolution"]),
                                    self.encode(params["net_resolution"]),
                                    self.encode(params["model_pose"]),
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],                    
                                    self.encode(params["default_model_folder"]))

    def __del__(self):
        """
        OpenPose Destructor: Destroys OpenPose object
        """
        self._libop.delOP(self.op)

    def forward(self, image, display = False):
        """
        Forward: Takes in an image and returns the human 2D poses, along with drawn image if required

        Parameters
        ----------
        image : color image of type ndarray
        display : If set to true, we return both the pose and an annotated image for visualization

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(3),dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, displayImage, display)
        array = np.zeros(shape=(size),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        if display:
            return array, displayImage
        return array

    def poseFromHM(self, image, hm, ratios=[1]):
        """
        Pose From Heatmap: Takes in an image, computed heatmaps, and require scales and computes pose

        Parameters
        ----------
        image : color image of type ndarray
        hm : heatmap of type ndarray with heatmaps and part affinity fields
        ratios : scaling ration if needed to fuse multiple scales

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        if len(ratios) != len(hm):
            raise Exception("Ratio shape mismatch")

        # Find largest
        hm_combine = np.zeros(shape=(len(hm), hm[0].shape[1], hm[0].shape[2], hm[0].shape[3]),dtype=np.float32)
        i=0
        for h in hm:
           hm_combine[i,:,0:h.shape[2],0:h.shape[3]] = h
           i+=1
        hm = hm_combine

        ratios = np.array(ratios,dtype=np.float32)

        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]

        self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size, ratios)
        array = np.zeros(shape=(size[0],size[1],size[2]),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        return array, displayImage

    @staticmethod
    def process_frames(frame, boxsize = 368, scales = [1]):
        base_net_res = None
        imagesForNet = []
        imagesOrig = []
        for idx, scale in enumerate(scales):
            # Calculate net resolution (width, height)
            if idx == 0:
                net_res = (16 * int((boxsize * frame.shape[1] / float(frame.shape[0]) / 16) + 0.5), boxsize)
                base_net_res = net_res
            else:
                net_res = ((min(base_net_res[0], max(1, int((base_net_res[0] * scale)+0.5)/16*16))),
                          (min(base_net_res[1], max(1, int((base_net_res[1] * scale)+0.5)/16*16))))
            input_res = [frame.shape[1], frame.shape[0]]
            scale_factor = min((net_res[0] - 1) / float(input_res[0] - 1), (net_res[1] - 1) / float(input_res[1] - 1))
            warp_matrix = np.array([[scale_factor,0,0],
                                    [0,scale_factor,0]])
            if scale_factor != 1:
                imageForNet = cv2.warpAffine(frame, warp_matrix, net_res, flags=(cv2.INTER_AREA if scale_factor < 1. else cv2.INTER_CUBIC), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                imageForNet = frame.copy()

            imageOrig = imageForNet.copy()
            imageForNet = imageForNet.astype(float)
            imageForNet = imageForNet/256. - 0.5
            imageForNet = np.transpose(imageForNet, (2,0,1))

            imagesForNet.append(imageForNet)
            imagesOrig.append(imageOrig)

        return imagesForNet, imagesOrig

    @staticmethod
    def draw_all(imageForNet, heatmaps, currIndex, div=4., norm=False):
        netDecreaseFactor = float(imageForNet.shape[0]) / float(heatmaps.shape[2]) # 8
        resized_heatmaps = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[1], imageForNet.shape[0], imageForNet.shape[1]))
        num_maps = heatmaps.shape[1]
        combined = None
        for i in range(0, num_maps):
            heatmap = heatmaps[0,i,:,:]
            resizedHeatmap = cv2.resize(heatmap, (0,0), fx=netDecreaseFactor, fy=netDecreaseFactor)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resizedHeatmap)

            if i==currIndex and currIndex >=0:
                resizedHeatmap = np.abs(resizedHeatmap)
                resizedHeatmap = (resizedHeatmap*255.).astype(dtype='uint8')
                im_color = cv2.applyColorMap(resizedHeatmap, cv2.COLORMAP_JET)
                resizedHeatmap = cv2.addWeighted(imageForNet, 1, im_color, 0.3, 0)
                cv2.circle(resizedHeatmap, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
                return resizedHeatmap
            else:
                resizedHeatmap = np.abs(resizedHeatmap)
                if combined is None:
                    combined = np.copy(resizedHeatmap);
                else:
                    if i <= num_maps-2:
                        combined += resizedHeatmap;
                        if norm:
                            combined = np.maximum(0, np.minimum(1, combined));

        if currIndex < 0:
            combined /= div
            combined = (combined*255.).astype(dtype='uint8')
            im_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(imageForNet, 0.5, im_color, 0.5, 0)
            cv2.circle(combined, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
            return combined

# ----------------------------------------------------------------------------
#WARNING: Does not work with the latest openpose version which contains 25 joint positions, not 18
joint_names = ['nose', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'midhip', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear', 'lbigtoe', 'lsmalltoe', 'lheel', 'rbigtoe', 'rsmalltoe', 'rheel']

'''
Assign each of the tuples to a name that indicates which of the joints that tuple represents.

@param keypoint: the keypoint list from the .yml file
@return: a list of tuple for the joint name and the joint values
'''
def keypoints_to_joint_names(keypoints):
    global joint_names
	
    joints = []
    try:
        for i in range(0, len(keypoints)):
            joints.append([joint_names[i], keypoints[i]])
        return joints
    except:
        #If there are no joints in the frame, return an empty joint list for 1 person
        joints = []
        print("No Joints")
        for i in range(0, 25):
            joints.append([joint_names[i], [0,0,0]])
        return joints

'''
Get the joint values from the keypoint list based on which joint name was given as a parameter

@param keypoints: the keypoint data from the .yml file
@param joint_name: the name of the joint that the user wants to get the x,y, score valuess for
@return: a tuple of [x,y,score] for the given joint name
'''	
def get_keypoint_by_name(keypoints, joint_name):
	joints = keypoints_to_joint_names(keypoints)
	for name in joints:
		if name[0] == joint_name:
			return name[1]
	print("JOINT NOT FOUND")
	return False

'''
Method that draws the demo program given by EHU, which was originally in the __main__ code.

@param img: the image that the system must alter/work with.
@return: image that the system must print in the window of the program as output.
'''
def demo_program(keypoints):
    for i in range(len(keypoints)):
        person = keypoints[i]
        right_shoulder = get_keypoint_by_name(person, "rshoulder")
        right_elbow = get_keypoint_by_name(person, "relbow")
        right_wrist = get_keypoint_by_name(person, "rwrist")
        print(f"Person {i}: shoulder x: {right_shoulder[0]}, shoulder y: {right_shoulder[1]}, shoulder confidence: {right_shoulder[2]}")
       
# ===Start of code made by Yoran Kerbusch======================================    
# ===CW1~T4 code===============================================================
# Possible draw modes. DRAW_NONE is the standard mode. Switching between modes can be done by putting your hands together.
DRAW_NONE = 1
DRAW_RECT = 2
DRAW_CIRCLE = 3
DRAW_ELIP = 4
# This variable dictates what shape is being drawn between a person's hands, if they are visible.
draw_mode = DRAW_NONE 
# Array to save all the shapes that have been saved by the person in this session.
shapes_drawn = []
# We don't want to add or remove shapes for every frame a person looks up or down. We want to do it for every time instead.
looking_up = False
looking_down = False
# So that we don't constantly switch for every frame the hands are together.
hands_were_together = False

'''
Method that determines if the two wrists of the given person are close enough to be together.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return boolean to tell the calling system if the person's hands are together or not.
'''
def hands_together(person, mm_per_pixel):
    l_wrist = get_keypoint_by_name(person, "lwrist")
    r_wrist = get_keypoint_by_name(person, "rwrist")
    
    if (isinstance(l_wrist, np.ndarray) and isinstance(r_wrist, np.ndarray)):
        x_difference = l_wrist[0] - r_wrist[0]
        y_difference = l_wrist[1] - r_wrist[1]
        
        # Divide the tolerance in mm by the amount of pixels per mm we calculated from the eyes.
        x_tolerance = 110 / mm_per_pixel
        # Tolerance on the y-acxis is smaller as we want to be sure the hands are flat together
        y_tolerance = 60 / mm_per_pixel
        
        # Hands are together if they are pressed flat to each other, causing the x-axis and y-axis to be close.
        return (((abs(x_difference) > 0.0) and (abs(x_difference) < x_tolerance))
                and ((abs(y_difference) > 0.0) and (abs(y_difference) < y_tolerance)))
        
'''
Helper method that checks if the given wrist is above the given shoulder, within a given tolerance
@author Yoran Kerbusch (24143341)

@param shoulder: np.ndarray that includes the x and y coordinates of the shoulder joint to check.
@param wrist: np.ndarray that includes the x and y coordinates of the wrist joint to check.
@return boolean to tell the calling system if the wrist is more than the tolerance above the shoulder.
'''
def check_if_hand_up(shoulder, wrist, y_tolerance):
    if (isinstance(shoulder, np.ndarray) and isinstance(wrist, np.ndarray)):
        # The hand is up if the position of it is at least more than the tolerance higher than the shoulder
        return ((wrist[1] > 0.0) and (wrist[1] < (shoulder[1] - y_tolerance)))
       
'''
Method that checks if the found person has their left arm up above the shoulder enough to be seen as having the hand up.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param y_tolerance: float is the amount the wrist has to be above the shoulder before being deemed as the hand being up.
@return boolean to tell the calling system if the hand is up or not.
'''
def left_hand_up(person, y_tolerance):
    l_shoulder = get_keypoint_by_name(person, "lshoulder")
    l_wrist = get_keypoint_by_name(person, "lwrist")
    return check_if_hand_up(l_shoulder, l_wrist, y_tolerance)
    
'''
Method that does the same as left_hand_up but for the right arm of the person.
@author Yoran Kerbusch (24143341)
'''
def right_hand_up(person, y_tolerance):
    r_shoulder = get_keypoint_by_name(person, "rshoulder")
    r_wrist = get_keypoint_by_name(person, "rwrist")
    return check_if_hand_up(r_shoulder, r_wrist, y_tolerance)
    
'''
Method that checks what hands are up and returns a string appropriate to that.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return String sentence that tells which hand (or both) is up, if any are in the first place.
'''
def hands_up(person, mm_per_pixel):
    y_tolerance = 150 / mm_per_pixel
    
    left_up = left_hand_up(person, y_tolerance)
    right_up = right_hand_up(person, y_tolerance)
    if (left_up and right_up):
        return " + hands up"
    elif (left_up):
        return " + l_hand up"
    elif (right_up):
        return " + r_hand up"
    return ""
    
'''
Helper method that checks if an arm of a person is being held horizontally (at 90 degrees, within tolerances).
@author Yoran Kerbusch (24143341)

@param shoulder: np.ndarray is the coordinates of the person's shoulder we want to compare the wrist and elbow to.
@param elbow: np.ndarray is the coordinates of the person's elbow.
@param wrist: np.ndarray is the coordinates of the person's wrist.
@return boolean that tells the calling system if the arm is being held horizontally.
'''
def check_if_arm_horizontal(shoulder, elbow, wrist, y_tolerance):
    if ((isinstance(shoulder, np.ndarray)) and (isinstance(elbow, np.ndarray)) and (isinstance(wrist, np.ndarray))):
        shoulder_elbow = shoulder[1] - elbow[1]
        elbow_wrist = elbow[1] - wrist[1]
        # We know the arm is (semi-)horizontal if the shoulder, elbow and wrist differ minimally in their y-axis location.
        return (((abs(shoulder_elbow) > 0.0) and (abs(shoulder_elbow) < y_tolerance))
                and ((abs(elbow_wrist) > 0.0) and (abs(elbow_wrist) < y_tolerance)))

'''
Method that checks if the person's left arm is being held horizontally (aka at 90 degrees when standing up).
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param y_tolerance: float is the tolerance at which the elbow and wrist can be up or down from the shoulder to still be deemed horizontal.
@return boolean that tells the calling system if the left arm is being held horizontally.
'''
def left_arm_horizontal(person, y_tolerance):
    l_shoulder = get_keypoint_by_name(person, "lshoulder")
    l_elbow = get_keypoint_by_name(person, "lelbow")
    l_wrist = get_keypoint_by_name(person, "lwrist")
    return check_if_arm_horizontal(l_shoulder, l_elbow, l_wrist, y_tolerance)
    
'''
Method that is the same as left_arm_horizontal, but for the right arm.
@author Yoran Kerbusch (24143341)
'''
def right_arm_horizontal(person, y_tolerance):
    r_shoulder = get_keypoint_by_name(person, "rshoulder")
    r_elbow = get_keypoint_by_name(person, "relbow")
    r_wrist = get_keypoint_by_name(person, "rwrist")
    return check_if_arm_horizontal(r_shoulder, r_elbow, r_wrist, y_tolerance)
        
'''
Method that returns a string that includes if the arms of the person (if any) are being held horizontally.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return String a message that include which (or both) arms are being held horizontally.
'''
def arms_horizontal(person, mm_per_pixel):
    y_tolerance = 25 / mm_per_pixel
    
    left_horizontal = left_arm_horizontal(person, y_tolerance)
    right_horizontal = right_arm_horizontal(person, y_tolerance)
    if (left_horizontal and right_horizontal):
        return " + arms horizontal"
    elif (left_horizontal):
        return " + l_arm horizontal"
    elif (right_horizontal):
        return " + r_arm horizontal"
    return ""

'''
Helper method that checks if the arm is being held straight at any angle, comparing the shoulder to the elbow and wrist.
@author Yoran Kerbusch (24143341)

@param shoulder: np.ndarray is the coordinates of the person's shoulder we want to compare the wrist and elbow to.
@param elbow: np.ndarray is the coordinates of the person's elbow.
@param wrist: np.ndarray is the coordinates of the person's wrist.
@return boolean that tells the calling system if the arm is being held straight, within tolerances.
'''
def check_if_arm_straight(shoulder, elbow, wrist, tolerance):
    if ((isinstance(shoulder, np.ndarray) and (shoulder[0] > 0.0))
        and (isinstance(elbow, np.ndarray) and (elbow[0] > 0.0))
        and (isinstance(wrist, np.ndarray) and (wrist[0] > 0.0))):
        angle_arm = math.degrees(math.atan2((shoulder[0] - wrist[0]), (shoulder[1] - wrist[1])))
        angle_elbow = math.degrees(math.atan2((shoulder[0] - elbow[0]), (shoulder[1] - elbow[1])))
        # Write the angle of the arm on the appropriate shoulder of the person, to give insight to the user.
        cv2.putText(img, str(round(angle_arm)), (shoulder[0], shoulder[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA, False)
        
        tolerance_up = angle_arm - tolerance
        tolerance_down =  angle_arm + tolerance
        return (angle_elbow > tolerance_up and angle_elbow < tolerance_down)

'''
Method that checks if the person is holding their arm straight, at any angle.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param tolerance: float the value within the elbow and wrist have to be angled from the shoulder to still be deemed as a straight held arm.
@return boolean that tells the calling system if the person is holding their left arm straight.
'''
def left_arm_straight(person, tolerance):
    l_shoulder = get_keypoint_by_name(person, "lshoulder")
    l_elbow = get_keypoint_by_name(person, "lelbow")
    l_wrist = get_keypoint_by_name(person, "lwrist")
    return check_if_arm_straight(l_shoulder, l_elbow, l_wrist, tolerance)

'''
Method that is the same as left_arm_straight, only for the right arm of the person.
@author Yoran Kerbusch (24143341)
'''
def right_arm_straight(person, tolerance):
    r_shoulder = get_keypoint_by_name(person, "rshoulder")
    r_elbow = get_keypoint_by_name(person, "relbow")
    r_wrist = get_keypoint_by_name(person, "rwrist")
    return check_if_arm_straight(r_shoulder, r_elbow, r_wrist, tolerance)

'''
Method that returns a String that holds which arm (if not both) are being held straight, for gesture recognition.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return String is a message telling what (if not both) arms are being held straight by the person.
'''
def arms_straight(person, mm_per_pixel):
    tolerance = 30 / mm_per_pixel
    
    left_straight = left_arm_straight(person, tolerance)
    right_straight = right_arm_straight(person, tolerance)
    if (left_straight and right_straight):
        return " + arms straight"
    elif (left_straight):
        return " + l_arm straight"
    elif (right_straight):
        return " + r_arm straight"
    return ""

'''
Method that collects all arm gestures the person is doing into one String, being displayed as one message.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return gestures: String a message that includes all arm gestures (if any) the person is doing.
'''
def arm_gestures(person, mm_per_pixel):
    gestures = ""
    
    # Check some points on the person's body to see what gesture they are making.
    # If they did a recognized gesture, update the last_gesture variable. Otherwise, do not update it.
    if hands_together(person, mm_per_pixel):
        gestures = gestures + " + hands together"
    
    gestures = gestures + arms_straight(person, mm_per_pixel)
    
    gestures = gestures + arms_horizontal(person, mm_per_pixel)
    
    gestures = gestures + hands_up(person, mm_per_pixel)
        
    if (len(gestures) < 1):
        gestures = " - none"
    
    return gestures

'''
Method that checks if the person is facing forwards or with their back to the camera.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@return String a message that tells the calling system what way the person is facing, if that could be determined.
'''
def head_forwards_or_backwards(person):
    l_shoulder = get_keypoint_by_name(person, "lshoulder")
    r_shoulder = get_keypoint_by_name(person, "rshoulder")
    if ((isinstance(l_shoulder, np.ndarray)) and (isinstance(r_shoulder, np.ndarray))):
        if ((l_shoulder[0] > r_shoulder[0])):
            # Determine if the person is facing forward by what shoulder is on which side of the x-axis.
            return " front"
        elif (l_shoulder[0] < r_shoulder[0]):
            # Simply check if the right shoulder is on the left side of the input. If so, the user is facing backwards.
            return " back"
    
    l_hip = get_keypoint_by_name(person, "lhip")
    r_hip = get_keypoint_by_name(person, "rhip")
    if ((isinstance(l_hip, np.ndarray)) and (isinstance(r_hip, np.ndarray))):
        # If we cannot see the shoulders, then maybe we can see the hips.
        if ((l_hip[0] > r_hip[0])):
            # Same as with the shoulders, depending on what way around the hips are, we know if the person is facing forwards.
            return " front"
        elif (l_hip[0] < r_hip[0]):
            return " back"
        
    l_ear = get_keypoint_by_name(person, "lear")
    r_ear = get_keypoint_by_name(person, "rear")
    nose = get_keypoint_by_name(person, "nose")
    if ((isinstance(nose, np.ndarray) and (isinstance(l_ear, np.ndarray) and (isinstance(r_ear, np.ndarray))))):
        # One final option if we can only see the head, we can see if the nose and at least one ear are visible.
        if (nose[0] > 0.0 and (l_ear[0] > 0.0 or r_ear[0] > 0.0)):
            # In that case, we assume the person is facing forwards.
            return " front"
    # If we cannot see anything useful, then return that we do not know.
    return " unknown"

'''
Method that checks if the person is looking up or down, as long as they are facing the camera enough to see and eye and ear.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return String a message that tells the calling system if the person is looking up, down or forwards.
'''
def facing_up_or_down(person, mm_per_pixel):
    l_ear = get_keypoint_by_name(person, "lear")
    r_ear = get_keypoint_by_name(person, "rear")
    nose = get_keypoint_by_name(person, "nose")
    
    if ((isinstance(l_ear, np.ndarray)) and (isinstance(r_ear, np.ndarray)) and (isinstance(nose, np.ndarray))):
        y_tolerance_up = 45 / mm_per_pixel
        if ((nose[1] < l_ear[1] - y_tolerance_up) or (nose[1] < r_ear[1] - y_tolerance_up)):
            # If the nose is above the ears + a certain amount of y-distance, the person is looking up.
            # We substract the distance, as the origin of the input is seen as the top-left.
            return " + up"
        
        y_tolerance_down = 25 / mm_per_pixel
        if (((l_ear[1] > 0.0) and (nose[1] > l_ear[1] + y_tolerance_down)) or ((r_ear[1] > 0.0) and (nose[1] > r_ear[1] + y_tolerance_down))):
            # If the person is looking down, the nose will be below the ears. We also check if either ear is visible.
            # This check has to be done, because otherwise an non-visible ear will always return [0, 0, 0], which the nose's coordinated will always be lower as, if the nose is visible.
            return " + down"
        # Else we assume the person is facing forwards straight.
        return " + forwards"

'''
Method that checks if the person is looking left or right, depending on how they are facing the camera.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param position: String is the message from facing_up_or_down, telling if the person is facing the camera with their front or back.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return String a message that tells the calling system if the person is looking left or right.
'''
def facing_left_or_right(person, position, mm_per_pixel):
    neck = get_keypoint_by_name(person, "neck")
    l_ear = get_keypoint_by_name(person, "lear")
    r_ear = get_keypoint_by_name(person, "rear")
    nose = get_keypoint_by_name(person, "nose")
    
    if ((isinstance(l_ear, np.ndarray)) and (isinstance(r_ear, np.ndarray)) and (isinstance(nose, np.ndarray))):
        y_tolerance = 40 / mm_per_pixel
        if ("front" in position):
            # The person is facing forwards, so when they look left, their nose points left.
            if ((nose[0] > neck[0] + y_tolerance)):
                # If the person has their nose a certain x-distance more to the left than the neck, they are facing left.
                return " + left"
            elif ((nose[0] < neck[0] - y_tolerance)):
                # The opposite is true for facing right then, of course.
                return " + right"
            # If they are within both boundaries, they are facing the camera (mostly) straight.
            return " + straight"
        elif ("back" in position):
            # The person is facing their back to us, so we need to use their ears to see if they are looking left or right.
            if (((l_ear[0] > 0.0) and (r_ear[0] < 0.0)) or (nose[0] < neck[0] - y_tolerance)):
                # If the left ear is visible but the right ear is not, the person is facing left. Or if their nose is on the right side.
                return " + left"
            elif (((r_ear[0] > 0.0) and (l_ear[0] < 0.0)) or (nose[0] > neck[0] + y_tolerance)):
                # The opposite counts for if they are facing right (which will be left for the camera as they have their back to it).
                return " + right"
            # If no ears or nose are visible, the person must have their head straight with the back to the camera.
            return " + straight"
        # We don't know the way the person is facing, so we cannot determine what way they are looking.
        return " + unknown"

'''
Method that collects a message on what way the person is looking, and how their body is facing the camera.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param mm_per_pixel: float amount of millimeters per pixel, to determine the tolerance.
@return position: String a message including the way the person's head is facing, to be displayed in the output.
'''
def head_position(person, mm_per_pixel):
    positions = ""
    
    # Determine if the person in view is facing the camera or not.
    positions = head_forwards_or_backwards(person)
        
    # Then also check if the person is looking up or down, or forwards
    positions = positions + facing_up_or_down(person, mm_per_pixel)
        
    # Finally, check if they are looking left or right.
    positions = positions + facing_left_or_right(person, positions, mm_per_pixel)
    
    return positions

'''
Method that determines the amount of millimeters per pixel the camera is seeing, allowing for rudimentary distance approximation.
This is assuming the person is an adult and has an average distance between their eyes of around 63 millimeters.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param avg_eye_distance: int is the average distance adults eyes are apart, which is usually 63 millimeters.
@return distance: float is the amount of millimeters each pixel represents, assuming from an average distance between adult eyes.
'''
def distance(person, avg_eye_distance):
    distance = ""
    
    l_eye = get_keypoint_by_name(person, "leye")
    r_eye = get_keypoint_by_name(person, "reye")
    if (l_eye[0] > 0.0 and r_eye[0] > 0.0):
        # Get the distance between the two eyes, if they are both visible.
        pixels_between_eyes = math.sqrt((r_eye[0] - l_eye[0])**2 + (r_eye[1] - l_eye[1])**2)
        # Then calculate how much distance each millimeter will be, assuming on the average distance.
        distance = avg_eye_distance / pixels_between_eyes
    else:
        distance = -1
    
    return distance

'''
Method that recognises gestures of one person on the camera & prints what it is.
For Interface Programming - Coursework 1, task 4
@author Yoran Kerbusch (EHU Student 24143341)

@param keypoints: array of recognised people from the input. This method only uses the first in the array.
@param img: input from the webcam (or any image) that needs to be scanned.
@return img: output for the window, including text stating the gestures being made.
'''
def pose_recognise(keypoints, img):
    avg_eye_distance = 63 # Average human adult eye distance in millimeters
    last_mm_per_pixel = 1.0
    last_gesture = ""
    last_head = ""
    
    global DRAW_SQUARE
    global DRAW_CIRCLE
    global DRAW_RECT
    global draw_mode
    global shapes_drawn
    global looking_up
    global looking_down
    global hands_were_together
    
    # DEBUG messages to check if the recognition of someone holding one of these positions is correct.
    #print("Hands were together: " + str(hands_were_together))
    #print("Person was looking up: " + str(looking_up))
    #print("Person was looking down: " + str(looking_down))
    
    if (len(keypoints) > 0):
        first_person = keypoints[0]
        
        # Get the amount of mm each pixel covers from the distance between the eyes.
        mm_per_pixel = distance(first_person, avg_eye_distance);
        if (mm_per_pixel > 0):
            # Only update how much distance each pixel is if both eyes are visible. 
            # If a person moves their back or side to the camera, we assume they
            #  do not move closer or further from the camera, until both eyes are visible again.
            last_mm_per_pixel = mm_per_pixel
    
        # Get a string with the arm gestures being done by the person.
        last_gesture = arm_gestures(first_person, last_mm_per_pixel)
    
        # Then also recognise the person's head movement.
        last_head = head_position(first_person, last_mm_per_pixel)
        
        
        
        if (("together" in last_gesture) and (hands_were_together == False)):
            # Go to the next draw mode, unless we are at the last mode, in which case we loop back to 1.
            hands_were_together = True
            draw_mode = (draw_mode + 1) % 5 # the last number in this calculation is the same as the number of draw modes plus one.
        
        # Draw shapes between the person's hands, depending on the drawing mode they selected.
        l_wrist = get_keypoint_by_name(first_person, "lwrist")
        r_wrist = get_keypoint_by_name(first_person, "rwrist")
        line_size = int(round((0.5 + (5 / mm_per_pixel))))
        if ((l_wrist[0] > 0.0) and (r_wrist[0] > 0.0)):
            # Only if both wrists are visible, draw the shapes (if a draw mode is also on).
            l_x = int(round(l_wrist[0]))
            l_y = int(round(l_wrist[1]))
            r_x = int(round(r_wrist[0]))
            r_y = int(round(r_wrist[1]))
            
            if (draw_mode == DRAW_RECT):
                cv2.rectangle(img, (l_x, l_y), (r_x, r_y), (255, 255, 255), line_size)
            
                if (("down" in last_head) and (looking_down == False)):
                    # Save the shape only if the person nods.
                    shapes_drawn.append([DRAW_RECT, l_wrist[0], l_wrist[1], r_wrist[0], r_wrist[1]])
                    looking_down = True
            elif (draw_mode == DRAW_CIRCLE):
                # Do the calculation for what the center point and radius for the circle should be.
                radius = int(round((math.sqrt((l_x - r_x)**2 + (l_y - r_y)**2)) / 2))
                center_x = int(round(l_x - radius))
                center_y = int(round(l_y))
                        
                cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), line_size)
                        
                if (("down" in last_head) and (looking_down == False)):
                    shapes_drawn.append([DRAW_CIRCLE, center_x, center_y, radius])
                    looking_down = True
            elif (draw_mode == DRAW_ELIP): 
                radius = int(round((math.sqrt((l_x - r_x)**2 + (l_y - r_y)**2)) / 2))
                center_x = int(round(l_x - radius))
                center_y = int(round(l_y))
                
                width = int(round((max(l_x, r_x) - min(l_x, r_x)) / 2))
                length = max(l_y, r_y) - min(l_y, r_y)
                
                cv2.ellipse(img, (center_x, center_y), (width, length), 0, 0, 360, (255, 255, 255), line_size)
            
                if (("down" in last_head) and (looking_down == False)):
                    shapes_drawn.append([DRAW_ELIP, center_x, center_y, width, length])
                    looking_down = True
                
        if ("together" not in last_gesture):
            hands_were_together = False
        
        if ("down" not in last_head):
            # Only if the person stops having their head down after saving the last shape, we can allow shapes to be saved again.
            looking_down = False
        
        if ((len(shapes_drawn) > 0) and ("up" in last_head) and (looking_up == False)):
            # If the person just looked up, remove the last shape they drew from the array, but only once per time they looked up.
            shapes_drawn.pop()
            looking_up = True
        elif (("up" not in last_head)):
            # Once the person is not looking up anymore, save that.
            looking_up = False
            
        for shape in shapes_drawn:
            # Draw all the shapes the person previously drew and saved by nodding.
            if (shape[0] == DRAW_RECT):
                cv2.rectangle(img, (shape[1], shape[2]), (shape[3], shape[4]), (255, 255, 255), line_size);
            elif (shape[0] == DRAW_CIRCLE):
                cv2.circle(img, (shape[1], shape[2]), shape[3], (255, 255, 255), line_size)
            elif (shape[0] == DRAW_ELIP):
                cv2.ellipse(img, (shape[1], shape[2]), (shape[3], shape[4]), 0, 0, 360, (255, 255, 255), line_size)
       
        
        
    # Say with text in the top-left corner what gesture is being made/was made last.
    cv2.rectangle(img, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(img, "Gesture:{}".format(last_gesture), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA, False)
    cv2.putText(img, "Head:{}".format(last_head), (0, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA, False)
    
    return img
# ===End of CW1~T4 code========================================================

'''
REQUIREMENTS (MUST HAVE NVIDIA GPU):
    CUDA 9.0
    CUDNN for CUDA 9.0
    CMake
    Visual Studio 2017 C++ development tools
    Visual Studio 2017 Windows SDK x.xx.15
Place openpose folder on root directory of D:/ drive (Temporary solution)
All scripts made must be placed in openpose/Release/python/openpose directory.

@author of all adaptations: Yoran Kerbusch (24143341)
'''
if __name__ == "__main__":
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = "../../../models/"
    openpose = OpenPose(params)

    cap = cv2.VideoCapture(0)

    while 1:
        success, img = cap.read()
        
        # Code to run the demo program supplied with the original file (not my work):
        #keypoints, img = openpose.forward(img, True)
        #demo_program(keypoints)
        
        # Code to run the gesture recogniser system, made by Yoran Kerbusch (CW1 - TASK 4):
        keypoints, img = openpose.forward(img, True)
        img = pose_recognise(keypoints, img)
        
        # Show the image we have been given.
        cv2.imshow("output", img)
        
        # If the user presses "esc" on their keyboard, only then close the window.
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
# ===End of code made by Yoran Kerbusch=======================================