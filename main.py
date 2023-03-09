from numpy                      import array, diag , zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, deg2rad, pi
from numpy.linalg               import norm,inv

from structs.datahub            import DataHub
from structs.cam                import Cam
from structs.point3D			import Point3D


from lib.transformations        import *
from lib.epipolar_geom          import EpipolarGeom
from lib.feature_extracter      import FeatureExtracter
from lib.feature_matcher      	import FeatureMatcher
from lib.frame					import FrameHandler
from lib.visualizer             import Visualizer
from lib.bundle_adjustment		import BundleAdjustment

import os
import json
import cv2
import time
import rospy



class VO:


    
	def __init__(self, camera_info, ros_info, params, inits):
		
		self.DataHub 			= DataHub(camera_info, ros_info, params, inits)

		# rospy.init_node(ros_info["NODE_NAME"])

		# self.Frame				= FrameHandler		(self.DataHub)
		self.EplipolarGeom 		= EpipolarGeom		(self.DataHub)
		self.FeatureExtracter	= FeatureExtracter	(self.DataHub)
		self.FeatureMatcher		= FeatureMatcher	(self.DataHub)
		self.Visualizer 		= Visualizer		(self.DataHub)
		self.BundleAdjuster		= BundleAdjustment	(self.DataHub)


	def run_test(self):

		cam_list = []
		cam_hist = []

		for i in range(0,9,1):

			img = cv2.imread("./frames/frame000"+str(i)+".jpg",0)
			img_undist = cv2.undistort(img,self.DataHub.K,self.DataHub.dist_coeff)

			cami = Cam(img_undist)

			cam_list.append(cami)
		
		cam_list[0].T_W2B = self.DataHub.T_W2B_init
		cam_list[0].T_B2W = inv(self.DataHub.T_W2B_init)

		cam_hist.append(cam_list[0])

		self.FeatureExtracter.extract_feature(cam_list[0])
		
		for i in range(1,9,1):

			cam_prev = cam_list[i-1]
			cam_curr = cam_list[i]

			self.FeatureExtracter.extract_feature(cam_curr)

			self.FeatureMatcher.match_feature(cam_prev,cam_curr)

			T_B12B2 , cam_prev, cam_curr = self.EplipolarGeom.track_pose(cam_prev,cam_curr)
			
			self.BundleAdjuster.run(cam_prev, cam_curr, T_B12B2)
			
			cam_hist.append(cam_curr)


		self.Visualizer.viz_points(cam_curr.T_B2W@cam_curr.train_points3D, cam_curr.train_intensity)
		self.Visualizer.viz_trajec(cam_hist)
		self.Visualizer.run()



	def run(self):


		while len(self.DataHub.frame_prev) == 0:

			print("waiting for image to subscribed ... ",end="\r")


		cam_curr = Cam(self.DataHub.frame_curr,self.DataHub.T_W2B_init)
		self.FeatureExtracter.extract_feature(cam_curr)

		cam_hist = []

		time.sleep(0.1)

		start = time.time()
		end = time.time()

		while end-start < 10:

			cam_prev = cam_curr

			cam_curr = Cam(self.DataHub.frame_curr)

			self.FeatureExtracter.extract_feature(cam_curr)

			self.FeatureMatcher.match_feature(cam_prev,cam_curr)

			T_B12B2 , cam_prev, cam_curr = self.EplipolarGeom.track_pose(cam_prev,cam_curr)

			self.BundleAdjuster.run(cam_prev, cam_curr, T_B12B2)

			# print(T_B12B2)
			cam_hist.append(cam_curr)

			end = time.time()
			time.sleep(1/30)

		# cv2.destroyAllWindows()


		self.Visualizer.viz_trajec(cam_hist)
		self.Visualizer.run()



if __name__ == "__main__":


	path = os.path.dirname( os.path.abspath( __file__ ) )

	with open(os.path.join(path,("input.json")),'r') as fp:
		inputs = json.load(fp)

		camera_info = inputs["camera_info"]
		ros_info	= inputs["ros_info"]
		params 		= inputs["params"]
		inits 		= inputs["init"]

		VO_ = VO(camera_info, ros_info, params, inits)

		VO_.run_test()