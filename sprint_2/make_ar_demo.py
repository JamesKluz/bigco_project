import argparse
import cv2
import imutils
import numpy as np
from scipy import ndimage
from skimage.measure import label
import time

BOTTOM_RIGHT = np.zeros((1074, 1894), dtype=bool)
BOTTOM_RIGHT[550:, 960:] = True
TOP_RIGHT = np.zeros((1074, 1894), dtype=bool)
TOP_RIGHT[:549, 960:] = True

def fake_ar():
  """ 
    Creates masked video for loaded video file and 
    saves it to self.output_file
    This method additionally displays the original video
    along side the masked one.
  """
  # loop over frames
  # a buffer of pixels to seperate the two videos in the output
  video_capture = cv2.VideoCapture(args.in_video)
  frame_count = 0
  # Get frame count in video
  try:
    cpfc =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
      else cv2.CAP_PROP_FRAME_COUNT
    frame_count = int(video_capture.get(cpfc))
    print("{} frames in {}".format(frame_count, args.in_video))
  except:
    print("Could not gte frame count for {}".format(args.in_video))
  print("Writing to {}".format(args.out_video))
  first_pass = True
  writer = None

  if args.bg_1:
    tr_image = cv2.imread(args.bg_1)
    tr_image = cv2.resize(tr_image,(934, 549))
  else:
    tr_image = np.zeros((549, 934, 3)) + (180, 105, 255)
  if args.bg_2:
    br_image = cv2.imread(args.bg_2)
    br_image = cv2.resize(br_image,(934, 524))
  else:
    br_image = np.zeros((524, 934, 3)) + (180, 105, 255)
  while True:
    start_time = time.time()
    # read the next frame from the file
    (success, frame) = video_capture.read()
    # if we didn't succeed we hit the end of the feed
    if not success:
      break

    # # We double the image height plus a buffer to output both vids
    # output = np.zeros((2*image.shape[0] + border, 
    #                    2*image.shape[1] + border, 3), dtype='uint8')
    # output[:, :, :] = np.array([0, 0, 0])
    # output[:image.shape[0], :image.shape[1], :] = image[:, :, :]
    # #output[image.shape[0] + border :, :image.shape[1], :] = image[:, :, :]
    # output[:image.shape[0], image.shape[1] + border :, :] = image[:, :, :]
    # #output[image.shape[0] + border :, image.shape[1] + border :, :] = image[:, :, :]
    # grey_image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
    # get_sky_mask()
    # image[final_mask] = np.array([255, 0, 0])
    # output[image.shape[0] + border :, :image.shape[1], :] = image[:, :, :]
    # #output[:image.shape[0], image.shape[1] + border :, :] = image[:, :, :]
    # output[image.shape[0] + border :, image.shape[1] + border :, :] = image[:, :, :]
    end_time = time.time()
    # mask = (frame[:, :, 0] > 245) & (frame[:, :, 1] < 5) & (frame[:, :, 2] < 5)
    # frame[mask] = (180, 105, 255)
    mask = (frame[:, :, 0] > 204) & (frame[:, :, 1] < 110) & (frame[:, :, 2] < 110)
    mask[1:, :] = np.logical_or(mask[1:, :] , mask[:-1, :])
    mask[:-1, :] = np.logical_or(mask[:-1, :] , mask[1:, :])
    mask[:, 1:] = np.logical_or(mask[:, 1:] , mask[:, :-1])
    mask[:, :-1] = np.logical_or(mask[:, :-1] , mask[:, 1:])
    # mask = np.logical_and(mask, TOP_RIGHT)
    
    # Top right
    frame[:549, 960:, :][mask[:549, 960:]] = tr_image[mask[:549, 960:]]

    # Bottom Left
    frame[550:, 960:, :][mask[550:, 960:]] = br_image[mask[550:, 960:]]

    # mask = (frame[:, :, 0] == 238) & (frame[:, :, 1] == 3) & (frame[:, :, 2] == 0)
    # frame[mask] = (180, 105, 255)
    # mask = (frame[:, :, 0] == 246) & (frame[:, :, 1] == 0) & (frame[:, :, 2] == 0)
    # frame[mask] = (180, 105, 255)
    # mask = (frame[:, :, 0] == 246) & (frame[:, :, 1] == 0) & (frame[:, :, 2] == 0)
    # frame[mask] = (180, 105, 255)
    # mask = (frame[:, :, 0] == 247) & (frame[:, :, 1] == 0) & (frame[:, :, 2] == 0)
    # frame[mask] = (180, 105, 255)

    # check if the video writer is None
    if first_pass:
      first_pass = False
      # initialize our video writer
      # vw_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
      vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
      # we double the height and add a little 
      writer = cv2.VideoWriter(args.out_video, vw_fourcc, 30,
                              (frame.shape[1], frame.shape[0]), True)
      # Display processing time
      if frame_count > 0:
        elapsed_time = (end_time - start_time)
        print("Per-frame time: {:.4f} seconds".format(elapsed_time))
        print("Estimated total time: {:.4f}".format(
              elapsed_time * frame_count))

    # write the output frame to disk
    writer.write(frame)

    # Display video
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If `q` key pressed, break.
    if key == ord("q"):
      break

  writer.release()
  video_capture.release()

def main():
  fake_ar()

def parse_args():
  parser = argparse.ArgumentParser(
      description="Tool for 'faking' AR for sprint 2"
  )
  parser.add_argument("in_video", help="Path to input file")
  parser.add_argument("out_video", help="Path to input file")
  parser.add_argument("--bg_1", help="Path to input file")
  parser.add_argument("--bg_2", help="Path to input file")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main()