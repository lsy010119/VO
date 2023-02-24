from numpy                      import array, deg2rad, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg               import norm,inv

from structs.datahub                import DataHub
from structs.cam                    import Cam
from lib.transformations        import *
from lib.epipolar_geom          import EpipolarGeom
from visualizer                 import Visualizer

import argparse
import os
import json



class VO:


    
	def __init__(self, camera_info, params, inits):
		
		DataHub_ = DataHub(camera_info, params, inits)
		
		# self.Visualizer = Visualizer(DataHub_)

















if __name__ == "__main__":


	path = os.path.dirname( os.path.abspath( __file__ ) )

	with open(os.path.join(path,("input.json")),'r') as fp:
		inputs = json.load(fp)

		camera_info = inputs["camera_info"]
		params 		= inputs["params"]
		inits 		= inputs["init"]

		VO_ = VO(camera_info, params, inits)

		# VO_.run()