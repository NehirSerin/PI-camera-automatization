from pipython import GCSDevice, pitools
from pypylon import pylon
import os
import lib.circle_detection as cdt
import cv2
import numpy as np
from time import sleep 

# CONSTANTS
CONTROLLERNAME = "C-884.DB"
STAGES = ["M-126.PD2", "M-126.PD2", "M-126.PD2", "NOSTAGE"]
REFMODES = ["FNL", "FRF"]
SERIALNUM = "0000000000"

DY, DX = 0.3, 0.3
VERTEX = {"0,0": (7.0, 7.0), "1,1": (15.0, 13.0)}
STEP_NUM = [
    int(abs(VERTEX["0,0"][0] - VERTEX["1,1"][0]) // DX),
    int(abs(VERTEX["0,0"][1] - VERTEX["1,1"][1]) // DY),
]

AXES = {"x": 1, "y": 3, "z": 2}
DIR = "asdf"

def main():
    pidevice = connect_pi(CONTROLLERNAME, SERIALNUM, STAGES, REFMODES)
    camera = connect_camera(500, 40)

    get_kernels(pidevice, camera, DIR, 512, 512)

    pidevice.CloseConnection()
    camera.Close()


# CONNECT DEVICES
def connect_pi(controllername, serialnum, stages, refmodes):
    pidevice = GCSDevice(controllername)
    pidevice.ConnectUSB(serialnum=serialnum)
    pitools.startup(pidevice, stages=stages, refmodes=refmodes)
    return pidevice


def connect_camera(buffer_val, exposure):
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.MaxNumBuffer.Value = buffer_val
    camera.ExposureTime.Value = exposure
    return camera



def capture_single_image(camera):
    # Start grabbing a single image
    camera.StartGrabbingMax(1)
    
    while camera.IsGrabbing():
        with camera.RetrieveResult(2000) as result:
            if result.GrabSucceeded():
                # Convert the image to pylon image
                image = pylon.PylonImage()
                image.AttachGrabResultBuffer(result)
                
                # Use the ImageFormatConverter to convert the image to grayscale
                converter = pylon.ImageFormatConverter()
                converter.OutputPixelFormat = pylon.PixelType_Mono8  # Convert to grayscale (Mono8)
                converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                
                # Convert the image
                grayscale_image = converter.Convert(image)
                
                # Convert the grayscale image to np.ndarray
                image_array = grayscale_image.Array
                
                return np.asarray(image_array)
            
    camera.StopGrabbing()


def get_kernels(pidevice, camera, dir, width, height):
    
    os.makedirs(dir, exist_ok=True)  # Ensure directory exists
    
    org_width = camera.Width.Value
    org_height = camera.Height.Value
    
    

    for y in range(STEP_NUM[1] + 1):
        for x in range(STEP_NUM[0] + 1) if y % 2 == 0 else range(STEP_NUM[0] + 1)[::-1]:
            pidevice.MOV(AXES["x"], VERTEX["0,0"][0] + x * DX)
            pidevice.MOV(AXES["y"], VERTEX["0,0"][1] + y * DY)
            
            pitools.waitontarget(pidevice, axes=(AXES["x"], AXES["y"]))

            org_image = capture_single_image(camera)                
            
            

            circles = cdt.get_circle(org_image)
            os.makedirs(os.path.join(dir, f"frame{x}_{y}"), exist_ok=True)  # Fixed path

            for idx, circle in enumerate(circles):
                x_c, y_c, r_c = circle

                camera.Width.Value = width
                camera.Height.Value = height
                camera.OffsetX.Value = max(0, x_c - width // 2)  # Fixed offset
                camera.OffsetY.Value = max(0, y_c - height // 2)

                move_to_focus(pidevice, camera)
                
                camera.StartGrabbingMax(10)

                while camera.IsGrabbing():
                    with camera.RetrieveResult(2000) as result:
                        if result.GrabSucceeded():
                            img = pylon.PylonImage()
                            img.AttachGrabResultBuffer(result)

                            filename = os.path.join(dir, f"frame{x}{y}", f"image{idx}.tiff")
                            img.Save(pylon.ImageFileFormat_Tiff, filename)

                            img.Release()  # Free buffer inside 'with' block

                camera.StopGrabbing()
            
            camera.OffsetX.Value = 0
            camera.OffsetY.Value = 0
            camera.Width.Value = org_width
            camera.Height.Value = org_height
    
    
def move_to_focus(pidevice, camera, dz=0.005):
    """
    Function to find the sharpest image by moving the camera along the z-axis 
    and calculating the sharpness based on edge detection (Canny).

    Parameters:
    - pidevice: The device object to control the stage.
    - camera: The camera object to capture images.
    - dz: The step size for movement along the z-axis (default is 0.005).

    Returns:
    - best_focus: The z-axis position where the sharpest image was found.
    """
    sharpness_scores = []
    step_nums = np.arange(-10, 11)  # Step range from -10 to 10, inclusive

    # Get current z position
    current_z = pidevice.qPOS(AXES["z"])
    
    for step_num in step_nums:
        # Move the stage along the z-axis
        target_z = current_z + dz * step_num
        pidevice.MOV(AXES["z"], target_z)
        
        # Capture the image at the current z position
        img = capture_single_image(camera)
        
        # Ensure the image is grayscale (needed for Canny)
        if len(img.shape) == 3:  # If it's not grayscale, convert it
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection to find sharpness
        edges = cv2.Canny(img, threshold1=100, threshold2=200)
        sharpness = np.sum(edges)  # Sum of edge pixel intensities
        sharpness_scores.append(sharpness)

    # Find the index of the maximum sharpness score
    best_index = np.argmax(sharpness_scores)
    best_focus = current_z + dz * step_nums[best_index]
    
    pidevice.MOV(AXES["z"], best_focus)  # Move to the best focus position    


if __name__ == "__main__":
    main()
