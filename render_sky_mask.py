import argparse
import cv2
import numpy as np
import os
from SkySegmenter import SkySegmenter 
import sys
import time

MODEL_PATH = "resources/enet-model.net"

def set_model_parameters(model):
  if args.image_width is not None:
    model.width = args.image_width
  if args.sharpening_sigma is not None:
    model.sharpening_sigma = args.sharpening_sigma
  if args.no_sharpen:
    model.apply_sharpening = False
  if args.use_color:
    model.use_color_assertions = True
  if args.binary_thresh:
    model.binary_lower_bound = args.binary_thresh
  if args.no_normalize:
    model.normalize_grey = False
  if args.no_use_cc:
    model.use_connected_components = False
  if args.no_use_grey:
    model.use_greyscale = False
  if args.b4_and_after:
    model.b4_and_after = True

def main():
  ss = SkySegmenter()
  ss.load_cnn(args.enet_model)
  set_model_parameters(ss)
  if args.video:
    if args.output is None:
      print "Must specify output for video"
      sys.exit(1)
    ss.load_video(args.input, args.output)
    ss.generate_masked_video()
  else:
    ss.load_image(args.input)
    mask = ss.get_sky_mask()
    masked_image = np.array(ss.image)
    masked_image[mask] = np.array([255, 0, 0])
    cv2.imshow("Input", ss.image)
    cv2.imshow("Output", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
def parse_args():
  parser = argparse.ArgumentParser(
      description="Sky segmentation script for images and videos"
  )
  parser.add_argument("input", help="Path to input file")
  parser.add_argument("--output", help="Path to output file (optional)")
  parser.add_argument("--enet_model", default=MODEL_PATH, help="Path to CNN")
  parser.add_argument("--video", help="set this flag for video input", 
                      action="store_true")
  parser.add_argument("--image_width", type=int, help="image out width")
  parser.add_argument("--binary_thresh", type=int, 
                      help="lower bound for connected components")
  parser.add_argument("--sharpening_sigma", type=float, 
                      help="sigma for greyscale sharpening")
  parser.add_argument("--no_sharpen", help="don't sharpen greyscale image", 
                      action="store_true")
  parser.add_argument("--use_color", help="Use color check", 
                      action="store_true")
  parser.add_argument("--no_normalize", help="don't normalize grey image", 
                      action="store_true")
  parser.add_argument("--no_use_cc", help="don't use connected components", 
                      action="store_true")
  parser.add_argument("--no_use_grey", help="don't use connected components", 
                      action="store_true")
  parser.add_argument("--b4_and_after", help="render before and after videos", 
                      action="store_true")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main()