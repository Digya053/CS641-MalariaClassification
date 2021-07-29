import os
import cv2

from tqdm import tqdm
from pathlib import Path

def create_data(datadir, classes, img_size):
    """Creates dataset by taking images from folder and labelling the data
    with the directory it is in. Also, resizes the image to convert all the    images to the same size.
    
    Parameters
    ----------
    datadir: str
	The path to the data directory
    classes: list
	Target names
    img_size: int
	The size to resize an image into

    Returns
    -------
    data: list
        The created dataset
    """
    data = []
    for category in classes: 
        # path to the data directory
        path = os.path.join(datadir, category) 
        class_num = classes.index(category) 

        for img in tqdm(os.listdir(path)):
            try:
		# read images and resize it
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
                new_array = cv2.resize(img_array, (img_size, img_size), interpolation = cv2.INTER_CUBIC) 
                data.append([new_array, category])
            except Exception as e:
                pass
    return data
