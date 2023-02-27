from numpy        import array, deg2rad, zeros, ones, eye, block, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv,svd
import cv2

import matplotlib.pyplot as plt

class FeatureMatcher:



    def __init__(self, DataHub):
        
        self.DataHub = DataHub

        self.matcher = cv2.BFMatcher_create(normType=cv2.NORM_L1,crossCheck=True)
        

    def match_feature(self, cam1, cam2):

        matches = self.matcher.match(cam1.desc, cam2.desc)

        if len(matches) > self.DataHub.PARAM_mtchth: N_match = self.DataHub.PARAM_mtchth 
        else: N_match = len(matches)

        matches = sorted(matches, key = lambda x : x.distance)[:N_match]

        points2D_1      = ones((3,N_match))
        points2D_2      = ones((3,N_match))
        matched_idx     = [0]*N_match
        intensity_2     = zeros(N_match)

        for idx, match in enumerate(matches):

            point_matched_1     = cam1.keypoints[match.queryIdx]
            point_matched_2     = cam2.keypoints[match.trainIdx]


            points2D_1[0,idx]   = point_matched_1.pt[0]
            points2D_1[1,idx]   = point_matched_1.pt[1]

            points2D_2[0,idx]   = point_matched_2.pt[0]
            points2D_2[1,idx]   = point_matched_2.pt[1]

            matched_idx[idx]    = (match.queryIdx, match.trainIdx)

            intensity_2[idx]    = cam2.img[int(point_matched_2.pt[1]),int(point_matched_2.pt[0])]
        
        cam1.points2D   = points2D_1
        cam2.points2D   = points2D_2   
        cam2.intensity  = intensity_2
        
        self.DataHub.matchidx = matched_idx