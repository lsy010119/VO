from numpy                      import array, deg2rad, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg               import norm,inv

from structs.datahub            import DataHub
from structs.cam                import Cam
from structs.point3D			import Point3D


from lib.transformations        import *
from lib.epipolar_geom          import EpipolarGeom
from lib.feature_extracter      import FeatureExtracter
from lib.feature_matcher      	import FeatureMatcher

from visualizer                 import Visualizer

import os
import json
import cv2
import time


class VO:


    
	def __init__(self, camera_info, params, inits):
		
		self.DataHub 			= DataHub(camera_info, params, inits)

		self.EplipolarGeom 		= EpipolarGeom(self.DataHub)
		self.FeatureExtracter	= FeatureExtracter(self.DataHub)
		self.FeatureMatcher		= FeatureMatcher(self.DataHub)
		self.Visualizer 		= Visualizer(self.DataHub)


	def run(self):

		frame1_ = "frame0001.jpg"
		frame2_ = "frame0002.jpg"
		frame3_ = "frame0003.jpg"

		frame1 = cv2.imread(frame1_,0)
		frame2 = cv2.imread(frame2_,0)
		frame3 = cv2.imread(frame3_,0)

		cam1 = Cam(frame1,self.DataHub.T_W2B_init)
		cam2 = Cam(frame2)
		cam3 = Cam(frame3)

		self.FeatureExtracter.extract_feature(cam1)
		self.FeatureExtracter.extract_feature(cam2)
		# self.FeatureExtracter.extract_feature(cam3)

		self.FeatureMatcher.match_feature(cam1,cam2)

		T_B12B2 , cam1, cam2 = self.EplipolarGeom.track_pose(cam1,cam2)

		# self.FeatureMatcher.match_feature(cam2,cam3)

		# T_B12B2 , cam2, cam3 = self.EplipolarGeom.track_pose(cam2,cam3)

		self.Visualizer.viz_pose(cam1.T_B2W)
		self.Visualizer.viz_pose(cam2.T_B2W)
		# self.Visualizer.viz_pose(cam3.T_B2W)
		self.Visualizer.viz_points(cam2.T_B2W@cam2.points3D, cam2.intensity)
		# self.Visualizer.viz_points(cam3.T_B2W@cam3.points3D, cam3.intensity)

		self.Visualizer.run()



if __name__ == "__main__":


	path = os.path.dirname( os.path.abspath( __file__ ) )

	with open(os.path.join(path,("input.json")),'r') as fp:
		inputs = json.load(fp)

		camera_info = inputs["camera_info"]
		params 		= inputs["params"]
		inits 		= inputs["init"]

		VO_ = VO(camera_info, params, inits)

		VO_.run()