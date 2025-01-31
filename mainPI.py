from pipython import GCSDevice, pitools
from pypylon import pylon
import numpy as np
import cv2 as cv2
from time import sleep

# TODO: It may be optimal to do it this way if implemented with notebook.
# CONNECTION -> IN A LOOP MOVE AND CAPTURE -> FOCUS TO OPTIMAL AND SAVE THE BUFFER -> END CONNECTION

# CONSTANTS
CONTROLLERNAME = "C-884.DB"  # 'C-884' will also work
STAGES = ["M-126.PD2", "M-126.PD2", "M-126.PD2", "NOSTAGE"]
REFMODES = ["FNL", "FRF"]
SERIALNUM = "0000000000"

DY, DX = 1, 1
VERTEX = {"0,0": (0, 0), "1,1": (1, 1)}

AXES = {"x": 1, "y": 3, "z": 2}

STEP_NUM = (1, 1)
STEP_SIZE = (1, 1)


def main():
    pidevice = connect_pi(CONTROLLERNAME, SERIALNUM, STAGES, REFMODES)
    camera = connect_camera(50)

    get_kernels(pidevice, camera)
    save_buffer_to_tiff_with_focus(camera, pidevice)

    pidevice.CloseConnection()
    camera.Close()


# CONNECTION AND SETUP OF DEVICES
def connect_pi(controllername, serialnum, stages, refmodes):
    pidevice = GCSDevice(controllername)
    pidevice.ConnectUSB(serialnum=serialnum)
    pitools.startup(pidevice, stages=stages, refmodes=refmodes)
    return pidevice
 

def connect_camera(buffer_val):
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.MaxNumBuffer.Value = buffer_val
    return camera


# MOVE PI AND CAPTURE IMAGES IN A LOOP
def get_kernels(pidevice, camera):
    pos_x, pos_y = pidevice.qPOS(AXES["x"]), pidevice.qPOS(AXES["y"])
    STEP_NUM = [
        abs(VERTEX["0,0"][0] - VERTEX["1,1"][0]) // DX,
        abs(VERTEX["0,0"][1] - VERTEX["1,1"][1]) // DY,
    ]

    for y in range(STEP_NUM[1]):
        for x in range(STEP_NUM[0]) if y % 2 == 0 else range(STEP_NUM[0])[::-1]:
            pidevice.MOV(AXES["x"], pos_x + x * DX)
            pidevice.MOV(AXES["y"], pos_y + y * DY)

            pitools.waitontarget(pidevice, axes=(AXES["x"], AXES["y"]))
            camera.GrabOne(100)

# SAVE THE BUFFER TO FILE WITH OPTIMAL FOCUS

def save_buffer_to_tiff_with_focus(camera, pidevice, z_range=(0.0, 1.0), z_step=0.1): # Output_dir parameter might be efficient for some implementations
    """
    Save captured images to TIFF files with focus adjustment before saving each image.
    :param camera: pylon.InstantCamera object
    :param pidevice: GCSDevice controlling the stage
    :param z_range: Tuple indicating range of z positions to scan (relative to current position)
    :param z_step: Step size for z scanning
    
    """
    img = pylon.PylonImage()
    for i in range(STEP_NUM[0] * STEP_NUM[1]):
        optimal_z = autofocus(pidevice, camera, z_range, z_step)
        pidevice.MOV(AXES["z"], optimal_z)
        pitools.waitontarget(pidevice, axes=(AXES["z"],))

        with camera.RetrieveResult(1200) as result: # Timeout value might be adjusted, inital was 2000 but for 1000 fps 1200 is enough.
            img.AttachGrabResultBuffer(result)

            filename = f"{i + 1}.tiff"
            img.Save(pylon.ImageFileFormat_Tiff, filename)

            img.Release() 
            
# FIND THE OPTIMAL Z-FOCUS

def autofocus(pidevice, camera, z_step=0.005):
    """
    Adjust focus by scanning through z positions and maximizing sharpness.
    :param pidevice: GCSDevice object controlling the stage
    :param camera: pylon.InstantCamera object
    :param z_range: Tuple indicating range of z positions to scan (relative to current position)
    :param z_step: Step size for z scanning
    :return: Optimal z position
    
    """
    current_z = pidevice.qPOS(AXES["z"])[AXES["z"]]
    z_range=(current_z - z_step*10, current_z + z_step*10)
    z_positions = np.arange(current_z + z_range[0], current_z + z_range[1], z_step)
    sharpness_scores = []

    for z in z_positions:
        pidevice.MOV(AXES["z"], z)
        pitools.waitontarget(pidevice, axes=(AXES["z"],))

        result = camera.GrabOne(100)
        img = result.GetArray()
  
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var() # Alternative: Sobel or Scharr operators, or Tenengrad
        
        sharpness_scores.append(sharpness)

    optimal_index = np.argmax(sharpness_scores)
    optimal_z = z_positions[optimal_index]

    return optimal_z

from pipython import GCSDevice, pitools
from pypylon import pylon
import cv2
import numpy as np
import os
import glob

def find_sharpest_image(fpath):
    
    folder_path = r"test"  # Make sure this path is correct
    image_paths = glob.glob(os.path.join(folder_path, "*.tiff"))

    # Loop through each image
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
           print(f"Warning: Could not read {img_path}")
           continue  # Skip the invalid image


    
    if not image_paths:
        print("No TIFF images found in the folder!")
        return None, None, None

    sharpness_scores = []
    images = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f" Warning: Could not read {img_path}")
            continue  

        images.append(img)
        edges = cv2.Canny(img, threshold1=100, threshold2=200)
        sharpness = np.sum(edges)

    # Calculate the sharpness score as the sum of the magnitudes

        sharpness_scores.append(sharpness)

    if not images:
        print("No valid images were loaded!")
        return None, None, None

    # Find the sharpest image
    best_index = np.argmax(sharpness_scores)
    best_image = images[best_index]
    best_image_name = os.path.basename(image_paths[best_index])
    best_score = sharpness_scores[best_index]

    print(f"\n Sharpest Image: {best_image_name} with Score = {best_score:.2f}")
    
    return best_image, best_image_name, best_score



# Find the sharpest image
folder_path = r"test"

sharpest_img, sharpest_name, sharpness_score = find_sharpest_image(folder_path)


if __name__ == "__main__":
    main()
