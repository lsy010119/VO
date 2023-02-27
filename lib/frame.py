from numpy              import frombuffer, uint8
from copy               import deepcopy
from sensor_msgs.msg    import Image

import rospy
import cv2



class FrameHandler:

    def __init__(self, DataHub):
        
        self.DataHub = DataHub
        
        rospy.Subscriber(DataHub.topic_name,Image,self.callback_img,queue_size=1)


    def callback_img(self,msg):

        try:

            img = frombuffer(msg.data, dtype=uint8).reshape(msg.height, msg.width, -1)

            try:
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            except: img_gray = img

            img_gray_undist = cv2.undistort(img_gray,self.DataHub.K,self.DataHub.dist_coeff)

            self.DataHub.frame_prev = deepcopy(self.DataHub.frame_curr)

            self.DataHub.frame_curr = img_gray_undist


        except: print("recieve failed")