from numpy                      import array, deg2rad, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg               import norm,inv

from lib.transformations        import *
from lib.datahub                import DataHub
from lib.cam                    import Cam
from lib.epipolar_geom          import EpipolarGeom
from visualizer                 import Visualizer

import argparse
import os
import json

class VO:


    
    def __init__(self, input_data):
        
        self.input_data



















if __name__ == "__main__":


	path = os.path.dirname( os.path.abspath( __file__ ) )

	with open(os.path.join(path,("input.json")),'r') as fp:
		inputs = json.load(fp)

    
    