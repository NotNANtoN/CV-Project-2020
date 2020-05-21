import sys
import os 
import time
from datetime import datetime

import cv2

from webcam import Webcam


def take_imgs(chessboard_size=(11,7), kSaveImageDeltaTime=1):
    sys.path.append("../")
    os.makedirs("./calib_images", exist_ok=True)
    camera_num = 0
    if len(sys.argv) == 2:
            camera_num = int(sys.argv[1])
    print('opening camera: ', camera_num)

    webcam = Webcam(camera_num)
    webcam.start()
    
    lastSaveTime = time.time()
 
    while True:
        
        # get image from webcam
        image = webcam.get_current_frame()
        if image is not None: 

            # check if pattern found
            ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), chessboard_size, None)
        
            if ret == True:     
                print('found chessboard')
                # save image
                filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.bmp'
                image_path="./calib_images/" + filename
                
                elapsedTimeSinceLastSave = time.time() - lastSaveTime
                do_save = elapsedTimeSinceLastSave > kSaveImageDeltaTime
                print(elapsedTimeSinceLastSave, kSaveImageDeltaTime)
                if do_save:
                    lastSaveTime = time.time()
                    print('saving file ', image_path)
                    cv2.imwrite(image_path, image)

                # draw the corners
                image = cv2.drawChessboardCorners(image, chessboard_size, corners, ret)                       

            cv2.imshow('camera', image)                

        else: 
            pass
            #print('empty image')                
                            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        
    #webcam.quit()            


if __name__ == "__main__":
    take_imgs()

