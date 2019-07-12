# IP-CourseworkOne
Code made by Yoran Kerbusch for Edge Hill University course CIS3149 Interface Programming 2018-2019 for coursework 1

This is a Python code coursework project that has students test and play with OpenCV and OpenPose body recognition.

WARNING: To use these files, you are required to do the following:
- Download OpenPose from https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases;
I RECOMMEND TO USE THE GPU VERSION (faster, more precize and less crash-prone), BUT ONLY IF YOU HAVE A POWERFUL GPU CARD IN YOUR LAPTOP/PC.
- If using the GPU version (which this project has been tested on), also make sure to install the appropriate version of NVIDIA CUDA on your computer;
- Once downloaded, go to the folder location on the disk and folder you downloaded OpenPose on "disk:/.../openpose/Release/python/openpose/";
- Drop the files from this repository into that folder;
- Open the files with your Python editor of choice (I made & tested this code with Spyder 3.3.3);
- Click run on the file of your choice to see it run.

This code was made for assignments given at Edge Hill University for second-year class CIS2160 Computer Graphics & Moddeling, during 2018-2019.

The final grade I received for this portfolio of code I made for the assignment is 95.0%, and it was originally completed by me at 7-3-2019 (7th of March, 2019).

Documentation for these files and coursework 1 as a whole is found at: https://docs.google.com/document/d/19NHYsl_9I6sQcy-2BWlXj1aAQhWD4esgxS9QYkEfXjc/edit?usp=sharing

I in particular recommend taking a look at assignment 4 ("openpose_cw1_task4_-_Yoran_Kerbusch.py"), as this includes a wide suite of features I made with body recognition interfaces, including:
- Gesture recognition for for instance arm angles, arms up, back to camera, facial directions and distances;
- Distance recognition by distance between eyes, assumed that the user is an average adult (as this code only works off a single webcam, thus does not know depth)
- Put hands together to draw different shapes on screen between the hands (cycle through available shapes by putting hands together again);
- Keep shapes on the screen by nodding down once;
- Remove most recently placed shape by looking up once (to delete multiple, nod once for every delete desired);
- Shape sizes are influenced by the distance the user is away from the camera.
