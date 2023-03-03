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
        query_points2D      = ones((3,N_match)) # 2D Point coordinations of prev frame matched with curr frame 
        train_points2D      = ones((3,N_match)) # 2D Point coordinations of curr frame matched with prev frame

        query_indices       = zeros(N_match)    # 2D Point indices of prev frame matched with curr frame
        train_indices       = zeros(N_match)    # 2D Point indices of curr frame matched with prev frame

        query_intensity     = zeros(N_match)    # intensity of points in prev frame
        train_intensity     = zeros(N_match)    # intensity of points in curr frame

        for idx, match in enumerate(matches):

            query_keypoint     = cam_prev.keypoints[match.queryIdx]
            train_keypoint     = cam_curr.keypoints[match.trainIdx]


            query_points2D[0,idx]   = query_keypoint.pt[0]
            query_points2D[1,idx]   = query_keypoint.pt[1]

            train_points2D[0,idx]   = train_keypoint.pt[0]
            train_points2D[1,idx]   = train_keypoint.pt[1]

            query_indices[idx]  = match.queryIdx
            train_indices[idx]  = match.trainIdx

            query_intensity[idx]    = cam_prev.img[int(query_keypoint.pt[1]),int(query_keypoint.pt[0])]
            train_intensity[idx]    = cam_curr.img[int(train_keypoint.pt[1]),int(train_keypoint.pt[0])]
        
        cam_prev.query_points2D   = query_points2D
        cam_curr.train_points2D   = train_points2D

        cam_prev.query_indices    = query_indices
        cam_curr.train_indices    = train_indices

        cam_prev.query_intensity  = query_intensity
        cam_curr.train_intensity  = train_intensity
        
        img3 = cv2.drawMatches(cam_prev.img, cam_prev.keypoints, cam_curr.img, cam_curr.keypoints, matches[:N_match], cam_curr.img)

        # cv2.imshow("A",img3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()