from numpy                      import array, deg2rad, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
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

import os
import json
import cv2
import time
import rospy



class VO:


    
	def __init__(self, camera_info, ros_info, params, inits):
		
		self.DataHub 			= DataHub(camera_info, ros_info, params, inits)

		rospy.init_node(ros_info["NODE_NAME"])

		self.Frame				= FrameHandler		(self.DataHub)
		self.EplipolarGeom 		= EpipolarGeom		(self.DataHub)
		self.FeatureExtracter	= FeatureExtracter	(self.DataHub)
		self.FeatureMatcher		= FeatureMatcher	(self.DataHub)
		self.Visualizer 		= Visualizer		(self.DataHub)


	def run(self):


		while len(self.DataHub.frame_prev) == 0:

			print("waiting for image to subscribed ... ",end="\r")


		cam_curr = Cam(self.DataHub.frame_curr,self.DataHub.T_W2B_init)
		self.FeatureExtracter.extract_feature(cam_curr)

		cam_hist = []

		time.sleep(0.1)

		# while not rospy.is_shutdown():
		start = time.time()
		end = time.time()

		while end-start < 50:

			cam_prev = cam_curr

			cam_curr = Cam(self.DataHub.frame_curr)

			self.FeatureExtracter.extract_feature(cam_curr)

			self.FeatureMatcher.match_feature(cam_prev,cam_curr)

			T_B12B2 , cam_prev, cam_curr = self.EplipolarGeom.track_pose(cam_prev,cam_curr)

			# cam_hist.append(cam_curr)

			print(T_B12B2)
			end = time.time()
			time.sleep(1/30)

		cv2.destroyAllWindows()


		# self.Visualizer.viz_trajec(cam_hist)
		# self.Visualizer.run()



if __name__ == "__main__":


	path = os.path.dirname( os.path.abspath( __file__ ) )

	with open(os.path.join(path,("input.json")),'r') as fp:
		inputs = json.load(fp)

		camera_info = inputs["camera_info"]
		ros_info	= inputs["ros_info"]
		params 		= inputs["params"]
		inits 		= inputs["init"]

		VO_ = VO(camera_info, ros_info, params, inits)

		VO_.run()