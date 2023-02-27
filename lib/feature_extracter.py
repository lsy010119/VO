from numpy        import array, deg2rad, zeros, ones, eye, block, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv,svd
import cv2



class FeatureExtracter:



    def __init__(self, DataHub):
        
        self.DataHub = DataHub

        self.extracter = cv2.SIFT_create(300,nOctaveLayers=3,contrastThreshold=0.09)
        

    def extract_feature(self, cam):
        
        cam.keypoints, cam.desc = self.extracter.detectAndCompute(cam.img,None)

        cam.triangulated_kp    = zeros(len(cam.keypoints))