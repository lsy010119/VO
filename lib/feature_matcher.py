from numpy        import array, deg2rad, zeros, ones, eye, block, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv,svd
import cv2

import matplotlib.pyplot as plt

class FeatureMatcher:



    def __init__(self, DataHub):
        
        self.DataHub = DataHub

        self.matcher = cv2.BFMatcher_create(normType=cv2.NORM_L1,crossCheck=True)
        

    def match_feature(self, cam_prev, cam_curr):
        '''
        ### Match Feature

        match the features of previous frame and current frame
        '''

        matches = self.matcher.match(cam_prev.desc, cam_curr.desc)

        ### Cliping match number ###
        if len(matches) > self.DataHub.PARAM_mtchth: N_match = self.DataHub.PARAM_mtchth 
        else: N_match = len(matches)

        ### Filtering Well-Matched points ###
        matches = sorted(matches, key = lambda x : x.distance)[:N_match]

        ### Initializing memories ###
        points2D_1      = ones((3,N_match)) # 2D Point coordinations of prev frame matched with curr frame 
        points2D_2      = ones((3,N_match)) # 2D Point coordinations of curr frame matched with prev frame

        tri_with_curr   = zeros(N_match)    # 2D Point indices of prev frame matched with curr frame
        tri_with_prev   = zeros(N_match)    # 2D Point indices of curr frame matched with prev frame

        intensity_1     = zeros(N_match)    # intensity of points in prev frame
        intensity_2     = zeros(N_match)    # intensity of points in curr frame

        for idx, match in enumerate(matches):

            point_matched_1     = cam_prev.keypoints[match.queryIdx]
            point_matched_2     = cam_curr.keypoints[match.trainIdx]


            points2D_1[0,idx]   = point_matched_1.pt[0]
            points2D_1[1,idx]   = point_matched_1.pt[1]

            points2D_2[0,idx]   = point_matched_2.pt[0]
            points2D_2[1,idx]   = point_matched_2.pt[1]

            tri_with_curr[idx]  = match.queryIdx
            tri_with_prev[idx]  = match.trainIdx

            intensity_1[idx]    = cam_prev.img[int(point_matched_1.pt[1]),int(point_matched_1.pt[0])]
            intensity_2[idx]    = cam_curr.img[int(point_matched_2.pt[1]),int(point_matched_2.pt[0])]
        
        cam_prev.points2D_with_curr   = points2D_1
        cam_curr.points2D_with_prev   = points2D_2

        cam_prev.tri_with_curr        = tri_with_curr
        cam_curr.tri_with_prev        = tri_with_prev

        cam_prev.intensity_with_curr  = intensity_1
        cam_curr.intensity_with_prev  = intensity_2
        
        # img3 = cv2.drawMatches(cam_prev.img, cam_prev.keypoints, cam_curr.img, cam_curr.keypoints, matches[:N_match], cam_curr.img)

        # cv2.imshow("A",img3)
        # cv2.waitKey(1)