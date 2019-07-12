# -*- coding: utf-8 -*-
"""
Wrap the OpenPose library with Python.
To install run `make install` and library will be stored in /usr/local/python

TODO: NOTE FOR TEACHERS OR OTHERS USING THIS CODE:
    This file only includes the code for task 2 I (Yoran Kerbusch) made!
    All code for this task can be found in the bottom part of this file, as the top part is all OpenPose code.
"""

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
# ===CW1~T2 code===============================================================   
# Connections: array with each joint in the above array that is connected.
connections = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10], [10, 11], [11, 24], [12, 13], [13, 14], [14, 21], [15, 17], [16, 18], [19, 20], [21, 19], [21, 20], [22, 23], [24, 22], [24, 23]]
        
'''
Helper function to draw a circle on a given bodypart of a given person.
@author Yoran Kerbusch (EHU Student 24143341)
'''
def draw_circle(img, person, name, write_names = False):
    part = get_keypoint_by_name(person, name)
    if not (part[0] <= 0) and not (part[1] <= 0):
        # Draw a white dot on the given body part's x and y coordinates.
        cv2.circle(img, (part[0], part[1]), 6, (169, 169, 169), -1)
        
        if (write_names):
            #If the user wants it to, then names can be drawn on each point.
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(img, name, (part[0], part[1]), font, 1,(255,255,255), 1, cv2.LINE_AA)
            
        return part
    return False

'''
Helper function that draws a line from the first given joint to the second given joint.
@author Yoran Kerbusch (24143341)
'''
def draw_line(img, part1, part2):
    if isinstance(part1, np.ndarray) and isinstance(part2, np.ndarray):
        # Draw a white line between the (x,y) coordinates of the two given bodyparts.
        cv2.line(img, (part1[0], part1[1]), (part2[0], part2[1]), (169, 169, 169), 3)

'''
Method that draws a stickman on each person that is within the webcam's range.
For Interface Programming - Coursework 1, task 2
@author Yoran Kerbusch (EHU Student 24143341)

@param keypoints: the keypoint data from the .yml file, marking the people in the webcam's range.
@param people_to_draw: amount of people to draw, starting at person 0. MUST BE AT LEAST INT 1 AND NO MORE THAN len(keypoints). Can be left unspecified to use default of 5.
@return: img that the system must print in the window of the program as output.
'''
def stickman(keypoints, people_to_draw = 5, write_names = False):
    #img = np.zeros((480, 640, 3), np.uint8)
    
    global joint_names
    global connections
    
    # Swap the range(len(keypoints)) for 1 if you only want the first person to be drawn.
    for i in range(min(len(keypoints), people_to_draw)):
        person = keypoints[i]
        parts = []
        
        # Draw white dots (and joint names) on each of the joints
        for part in joint_names:
            bodypart = draw_circle(img, person, part, write_names)
            parts.append(bodypart)
            
        # Then draw each of the connection lines between each joint that should be connected.
        # This is done using an array of arrays, each connection inner-array including the numbers of the joints to be connected.
        for connection in connections:
            draw_line(img, parts[connection[0]], parts[connection[1]])
    
    return img
# ===End of CW1~T2 code========================================================

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
        
        # Code to run the real stickman program, made by Yoran Kerbusch (CW1 - TASK 2):
        keypoints, img = openpose.forward(img, True)
        # TODO: You can change to not show the names on each point by setting the last value to False.
        img = stickman(keypoints, 10, True)
        
        # Show the image we have been given.
        cv2.imshow("output", img)
        
        # If the user presses "esc" on their keyboard, only then close the window.
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
# ===End of code made by Yoran Kerbusch=======================================