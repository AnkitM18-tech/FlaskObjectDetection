#Importing packages
import os
from flask import Flask, render_template,request,url_for,redirect,send_from_directory
from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
import numpy as np
import urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_utils

#Setting model names and paths
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

#Detection
