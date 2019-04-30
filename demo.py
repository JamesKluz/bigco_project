import argparse
import cv2
import numpy as np
import os
import sys
import time
from SkySegmenterLight import SkySegmeterLight

def main():
    source = args.input
    ssl = SkySegmeterLight()
    if args.lock_left:
      ssl.lock_left = True
    ssl.videoSrc = source
    ssl.verbose = args.verbose
    ssl.fade = args.fade
    ssl.min_proportion_region = args.rp_pp
    ssl.high_speed = True
    ssl.hi_def = not args.no_hd
    files = [os.path.join('bg_images', x) for x in os.listdir('bg_images/') if x.endswith('jpeg')]
    files = sorted(files)
    ssl.background_list = files
    ssl.break_cc_w_edges = args.break_cc_w_edges
    if args.output is not None:
        ssl.videoOut = args.output
    ssl.demo()

def parse_args():
  parser = argparse.ArgumentParser(
      description="Sky segmentation script for images and videos"
  )
  parser.add_argument("--input", default="videos/demo.mp4", help="Path to input file")
  parser.add_argument("--lock_left", help="lock video to left", action="store_true")
  parser.add_argument("--output", help="Path to output file (optional)")
  parser.add_argument("--verbose", help="output info like fps", action="store_true")
  parser.add_argument("--background", help="path to background to paste on image/video")
  parser.add_argument("--fade", help="fade from real to background", action="store_true")
  parser.add_argument("--no_hd", help="dont output hi res video", action="store_true")
  parser.add_argument("--break_cc_w_edges", help="break connected components with edges", action="store_true")
  parser.add_argument("--rp_pp", type=float, default=0.01, help="region proposal proportion")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main()