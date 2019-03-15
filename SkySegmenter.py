import cv2
import imutils
import numpy as np
from scipy import ndimage
from skimage.measure import label
import time
import utilities

class SkySegmenter:
  def __init__(self):
    # Image / Frame variables:
    self.image = None
    self.output_file = None
    self.grey_image = None
    self.sky_index = 11
    self.cnn_model = None
    self.width = 500
    self.cnn_input_dims = (1024, 512)
    self.norm_coeffcient = 1.0/255.0
    self.sharpening_sigma = 2.0
    self.binary_lower_bound = 120
    self.binary_low_blue_bound = 160
    self.color_image_mask = None
    self.color_image_only = None
    self.grey_mask = None
    self.connected_components = None
    self.remaining_components = None
    self.color_assertion_mask = None
    # Options
    self.use_connected_components = True
    self.apply_sharpening = True
    self.use_greyscale = True
    self.vertical_flip = False
    # use_color_assertions has no effect currenty
    self.use_color_assertions = True
    self.normalize_grey = True
    # Video Variables
    self.video_capture = None
    self.writer = None
    self.frame_count = None
    self.fps = 30
    self.display_video = True
    self.b4_and_after = False
    self.four_way = False
    self.roll = 0
    self.portrait = False


  def load_image(self, image_file):
    """ 
      Loads image at path image_file as a numpy array
      @ image_file: path to image file to produce mask for
    """
    self.image = cv2.imread(image_file) 
    self.image = imutils.resize(self.image, width=self.width)
    self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)

  def load_video(self, video_file, output_file):
    """ 
      Loads pointer to video at path video_file.
      We require an output path for videos
      @ image_file: path to video file to produce mask for
    """
    # initialize the video capture pointer
    self.video_capture = cv2.VideoCapture(video_file)
    self.output_file = output_file
    # Get frame count in video
    try:
      cpfc =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
      self.frame_count = int(self.video_capture.get(cpfc))
      print("{} frames in {}".format(self.frame_count, video_file))
    except:
      print("Could not gte frame count for {}".format(video_file))
    print("Writing to {}".format(self.output_file))

  def ready_for_video(self):
    if self.video_capture is None:
      print("Video capture pointer not initialized")
      return False
    if self.cnn_model is None:
      print("CNN model not loaded")
      return False
    return True

  # def generate_masked_video(self):
  #   """ 
  #     Creates masked video for loaded video file and 
  #     saves it to self.output_file
  #   """
  #   # Checks for needed state:
  #   if not self.ready_for_video():
  #     return
  #   # If b4_and_after flag or four_way flag set we 
  #   # call the appropriate methods
  #   if self.four_way:
  #     self.generate_masked_video_4_way()
  #     return
  #   if self.b4_and_after:
  #     if self.portrait:
  #       self.generate_masked_video_b4_and_after_portrait()
  #     else:
  #       self.generate_masked_video_b4_and_after_horizontal()
  #     return
  #   # loop over frames
  #   while True:
  #     # read the next frame from the file
  #     (success, frame) = self.video_capture.read()
  #     # if we didn't succeed we hit the end of the feed
  #     if not success:
  #       break

  #     if self.vertical_flip:
  #       frame = np.flip(frame, axis=0)
  #       frame = np.flip(frame, axis=1)

  #     self.image = imutils.resize(frame, width=self.width)
  #     self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)
  #     start_time = time.time()
  #     self.get_sky_mask()
  #     self.image[self.final_mask] = np.array([255, 0, 0])
  #     end_time = time.time()
  #     # check if the video writer is None
  #     if self.writer is None:
  #       # initialize our video writer
  #       vw_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
  #       self.writer = cv2.VideoWriter(self.output_file, vw_fourcc, self.fps,
  #                                    (self.image.shape[1], self.image.shape[0]), 
  #                                     True)
  #       # Display processing time
  #       if self.frame_count > 0:
  #         elapsed_time = (end_time - start_time)
  #         print("Per-frame time: {:.4f} seconds".format(elapsed_time))
  #         print("Estimated total time: {:.4f}".format(
  #               elapsed_time * self.frame_count))

  #     # write the output frame to disk
  #     self.writer.write(self.image)

  #     # Display video
  #     if self.display_video:
  #       cv2.imshow("Frame", self.image)
  #       key = cv2.waitKey(1) & 0xFF
  #       # If `q` key pressed, break.
  #       if key == ord("q"):
  #         break

  #   # Release the pointers
  #   self.writer.release()
  #   self.video_capture.release()
  #   self.writer = None 
  #   self.video_capture = None

  def generate_masked_video(self):
    """ 
      Creates masked video for loaded video file and 
      saves it to self.output_file
    """
    # Checks for needed state:
    if not self.ready_for_video():
      return
    # If b4_and_after flag or four_way flag set we 
    # call the appropriate methods
    if self.four_way:
      self.generate_masked_video_4_way()
      return
    if self.b4_and_after:
      if self.portrait:
        self.generate_masked_video_b4_and_after_portrait()
      else:
        self.generate_masked_video_b4_and_after_horizontal()
      return
    # loop over frames
    tr_image = cv2.imread('background_images/aliens_6.jpg')
    tr_image = cv2.resize(tr_image,(935, 525))
    half_frame_count = 2.0 * float(self.frame_count) / (5.0)
    counter = 0
    # fade = 0.0
    # image = True
    fade = 1.0
    image = False
    while True:
      # read the next frame from the file
      (success, frame) = self.video_capture.read()
      # if we didn't succeed we hit the end of the feed
      if not success:
        break

      if self.vertical_flip:
        frame = np.flip(frame, axis=0)
        frame = np.flip(frame, axis=1)
      if counter > half_frame_count:
        image = False
        counter = 0
      counter += 1
      self.image = imutils.resize(frame, width=self.width)
      self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)
      start_time = time.time()
      if not image:
        self.get_sky_mask()
        fade += 0.023
        if fade < 1.0:
          self.image[self.final_mask] = fade * tr_image[self.final_mask] + (1.0 - fade) * self.image[self.final_mask]
        else:
          self.image[self.final_mask] = tr_image[self.final_mask]
      end_time = time.time()
      # check if the video writer is None
      if self.writer is None:
        # initialize our video writer
        vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        self.writer = cv2.VideoWriter(self.output_file, vw_fourcc, self.fps,
                                     (self.image.shape[1], self.image.shape[0]), 
                                      True)
        # Display processing time
        if self.frame_count > 0:
          elapsed_time = (end_time - start_time)
          print("Per-frame time: {:.4f} seconds".format(elapsed_time))
          print("Estimated total time: {:.4f}".format(
                elapsed_time * self.frame_count))

      # write the output frame to disk
      self.writer.write(self.image)

      # Display video
      if self.display_video:
        cv2.imshow("Frame", self.image)
        key = cv2.waitKey(1) & 0xFF
        # If `q` key pressed, break.
        if key == ord("q"):
          break

    # Release the pointers
    self.writer.release()
    self.video_capture.release()
    self.writer = None 
    self.video_capture = None

  def generate_masked_video_b4_and_after_horizontal(self):
    """ 
      Creates masked video for loaded video file and 
      saves it to self.output_file
      This method additionally displays the original video
      along side the masked one.
    """
    # loop over frames
    # a buffer of pixels to seperate the two videos in the output
    border = 25
    while True:
      # read the next frame from the file
      (success, frame) = self.video_capture.read()
      # if we didn't succeed we hit the end of the feed
      if not success:
        break

      if self.vertical_flip:
        frame = np.flip(frame, axis=0)
        frame = np.flip(frame, axis=1)

      self.image = imutils.resize(frame, width=self.width)
      # We double the image height plus a buffer to output both vids
      output = np.zeros((2*self.image.shape[0] + border, 
                         self.image.shape[1], 3), dtype='uint8')
      output[:, :, :] = np.array([240, 240, 240])
      output[self.image.shape[0] + border :, :, :] = self.image[:, :, :]
      self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)
      start_time = time.time()
      self.get_sky_mask()
      self.image[self.final_mask] = np.array([255, 0, 0])
      output[:self.image.shape[0], :, :] = self.image[:, :, :]
      end_time = time.time()
      # check if the video writer is None
      if self.writer is None:
        # initialize our video writer
        vw_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # we double the height and add a little 
        self.writer = cv2.VideoWriter(self.output_file, vw_fourcc, self.fps,
                                     (output.shape[1], output.shape[0]), True)
        # Display processing time
        if self.frame_count > 0:
          elapsed_time = (end_time - start_time)
          print("Per-frame time: {:.4f} seconds".format(elapsed_time))
          print("Estimated total time: {:.4f}".format(
                elapsed_time * self.frame_count))

      # write the output frame to disk
      self.writer.write(output)

      # Display video
      if self.display_video:
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF
        # If `q` key pressed, break.
        if key == ord("q"):
          break

    self.writer.release()
    self.video_capture.release()
    self.writer = None 
    self.video_capture = None

  def generate_masked_video_4_way(self):
    """ 
      Creates masked video for loaded video file and 
      saves it to self.output_file
      This method additionally displays the original video
      along side the masked one.
    """
    # loop over frames
    # a buffer of pixels to seperate the two videos in the output
    border = 25
    while True:
      # read the next frame from the file
      (success, frame) = self.video_capture.read()
      # if we didn't succeed we hit the end of the feed
      if not success:
        break

      if self.vertical_flip:
        frame = np.flip(frame, axis=0)
        frame = np.flip(frame, axis=1)

      self.image = imutils.resize(frame, width=self.width)
      # We double the image height plus a buffer to output both vids
      output = np.zeros((2*self.image.shape[0] + border, 
                         2*self.image.shape[1] + border, 3), dtype='uint8')
      output[:, :, :] = np.array([0, 0, 0])
      output[:self.image.shape[0], :self.image.shape[1], :] = self.image[:, :, :]
      #output[self.image.shape[0] + border :, :self.image.shape[1], :] = self.image[:, :, :]
      #output[:self.image.shape[0], self.image.shape[1] + border :, :] = self.image[:, :, :]
      #output[self.image.shape[0] + border :, self.image.shape[1] + border :, :] = self.image[:, :, :]
      self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)
      start_time = time.time()
      self.get_sky_mask()
      color_only = np.array(self.image)
      color_and_grey = np.array(self.image)
      self.image[self.final_mask] = np.array([255, 0, 0])
      color_only[self.color_image_only] = np.array([255, 0, 0])
      color_and_grey[self.color_image_mask] = np.array([255, 0, 0])

      output[self.image.shape[0] + border :, :self.image.shape[1], :] = color_only
      output[:self.image.shape[0], self.image.shape[1] + border :, :] = color_and_grey
      output[self.image.shape[0] + border :, self.image.shape[1] + border :, :] = self.image[:, :, :]
      end_time = time.time()
      # check if the video writer is None
      if self.writer is None:
        # initialize our video writer
        # vw_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        # we double the height and add a little 
        self.writer = cv2.VideoWriter(self.output_file, vw_fourcc, self.fps,
                                     (output.shape[1], output.shape[0]), True)
        # Display processing time
        if self.frame_count > 0:
          elapsed_time = (end_time - start_time)
          print("Per-frame time: {:.4f} seconds".format(elapsed_time))
          print("Estimated total time: {:.4f}".format(
                elapsed_time * self.frame_count))

      # write the output frame to disk
      self.writer.write(output)

      # Display video
      if self.display_video:
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF
        # If `q` key pressed, break.
        if key == ord("q"):
          break

    self.writer.release()
    self.video_capture.release()
    self.writer = None 
    self.video_capture = None

  # def generate_masked_video_4_way(self):
  #   """ 
  #     Creates masked video for loaded video file and 
  #     saves it to self.output_file
  #     This method additionally displays the original video
  #     along side the masked one.
  #   """
  #   # loop over frames
  #   # a buffer of pixels to seperate the two videos in the output
  #   border = 25

  #   tr_image = cv2.imread('background_images/storm_1.jpeg')
  #   tr_image = cv2.resize(tr_image,(935, 525))

  #   br_image = cv2.imread('background_images/night.jpg')
  #   br_image = cv2.resize(br_image,(935, 525))

  #   while True:
  #     # read the next frame from the file
  #     (success, frame) = self.video_capture.read()
  #     # if we didn't succeed we hit the end of the feed
  #     if not success:
  #       break

  #     if self.vertical_flip:
  #       frame = np.flip(frame, axis=0)
  #       frame = np.flip(frame, axis=1)

  #     self.image = imutils.resize(frame, width=self.width)
  #     top_right = np.array(self.image)
  #     bottom_right = np.array(self.image)
  #     # We double the image height plus a buffer to output both vids
  #     output = np.zeros((2*self.image.shape[0] + border, 
  #                        2*self.image.shape[1] + border, 3), dtype='uint8')
  #     output[:, :, :] = np.array([0, 0, 0])
  #     output[:self.image.shape[0], :self.image.shape[1], :] = self.image[:, :, :]
  #     #output[self.image.shape[0] + border :, :self.image.shape[1], :] = self.image[:, :, :]
  #     # output[:self.image.shape[0], self.image.shape[1] + border :, :] = self.image[:, :, :]
  #     #output[self.image.shape[0] + border :, self.image.shape[1] + border :, :] = self.image[:, :, :]
  #     self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)
  #     start_time = time.time()
  #     self.get_sky_mask()
  #     self.image[self.final_mask] = np.array([255, 0, 0])
  #     top_right[self.final_mask] = tr_image[self.final_mask]
  #     bottom_right[self.final_mask] = br_image[self.final_mask]
  #     output[self.image.shape[0] + border :, :self.image.shape[1], :] = self.image[:, :, :]
  #     #top right
  #     output[:self.image.shape[0], self.image.shape[1] + border :, :] = top_right
  #     output[self.image.shape[0] + border :, self.image.shape[1] + border :, :] = bottom_right
  #     end_time = time.time()
  #     # check if the video writer is None
  #     if self.writer is None:
  #       # initialize our video writer
  #       # vw_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
  #       vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
  #       # we double the height and add a little 
  #       self.writer = cv2.VideoWriter(self.output_file, vw_fourcc, self.fps,
  #                                    (output.shape[1], output.shape[0]), True)
  #       # Display processing time
  #       if self.frame_count > 0:
  #         elapsed_time = (end_time - start_time)
  #         print("Per-frame time: {:.4f} seconds".format(elapsed_time))
  #         print("Estimated total time: {:.4f}".format(
  #               elapsed_time * self.frame_count))

  #     # write the output frame to disk
  #     self.writer.write(output)

  #     # Display video
  #     if self.display_video:
  #       cv2.imshow("Frame", output)
  #       key = cv2.waitKey(1) & 0xFF
  #       # If `q` key pressed, break.
  #       if key == ord("q"):
  #         break

  #   self.writer.release()
  #   self.video_capture.release()
  #   self.writer = None 
  #   self.video_capture = None

  def generate_masked_video_b4_and_after_portrait(self):
    """ 
      Creates masked video for loaded video file and 
      saves it to self.output_file
      This method additionally displays the original video
      along side the masked one.
    """
    # loop over frames
    # a buffer of pixels to seperate the two videos in the output
    border = 25
    while True:
      # read the next frame from the file
      (success, frame) = self.video_capture.read()
      # if we didn't succeed we hit the end of the feed
      if not success:
        break

      if self.vertical_flip:
        frame = np.flip(frame, axis=0)
        frame = np.flip(frame, axis=1)
      if self.roll > 0:
        frame = np.rot90(frame, k=self.roll)

      self.image = imutils.resize(frame, width=self.width)
      # We double the image height plus a buffer to output both vids
      output = np.zeros((self.image.shape[0], 
                         2*self.image.shape[1] + border, 3), dtype='uint8')
      output[:, :, :] = np.array([240, 240, 240])
      output[:, self.image.shape[1] + border :, :] = self.image[:, :, :]
      self.grey_image = cv2.cvtColor(self.image ,cv2.COLOR_BGR2GRAY)
      start_time = time.time()
      self.get_sky_mask()
      self.image[self.final_mask] = np.array([255, 0, 0])
      output[:, :self.image.shape[1], :] = self.image[:, :, :]
      end_time = time.time()
      # check if the video writer is None
      if self.writer is None:
        # initialize our video writer
        vw_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # we double the height and add a little 
        self.writer = cv2.VideoWriter(self.output_file, vw_fourcc, self.fps,
                                     (output.shape[1], output.shape[0]), True)
        # Display processing time
        if self.frame_count > 0:
          elapsed_time = (end_time - start_time)
          print("Per-frame time: {:.4f} seconds".format(elapsed_time))
          print("Estimated total time: {:.4f}".format(
                elapsed_time * self.frame_count))

      # write the output frame to disk
      self.writer.write(output)

      # Display video
      if self.display_video:
        cv2.imshow("Frame", output)
        key = cv2.waitKey(1) & 0xFF
        # If `q` key pressed, break.
        if key == ord("q"):
          break

    self.writer.release()
    self.video_capture.release()
    self.writer = None 
    self.video_capture = None

  def load_cnn(self, cnn_path):
    self.cnn_model = cv2.dnn.readNet(cnn_path)

  def ready_for_image(self):
    if self.image is None:
      print("No image loaded")
      return False
    if self.cnn_model is None:
      print("CNN model not loaded")
      return False
    return True

  def get_sky_mask(self):
    """ 
      Runs the needed methods in order to generate the 
      full `sky/not-sky` mask
    """
    if not self.ready_for_image():
      return
    self.get_color_image_mask()
    self.get_color_assertion_mask()
    self.find_connected_components()
    if self.use_greyscale:
      self.get_grey_mask()
      self.color_image_mask = np.logical_or(self.color_image_mask, 
                                            self.grey_mask)
    self.get_final_mask()
    return self.final_mask
    

  def run_foward_pass(self, image):
    """ 
      Runs ENet on the image which returns per-pixel predictions
      for CityScape classes
    """
    blob = cv2.dnn.blobFromImage(image, self.norm_coeffcient, 
                                 self.cnn_input_dims, 0, swapRB=True, 
                                 crop=False)
    self.cnn_model.setInput(blob)
    output = self.cnn_model.forward()
    classMap = cv2.resize(np.argmax(output[0], axis=0), 
                          (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    mask = (classMap == self.sky_index)
    return mask

  def get_grey_mask(self):
    """ 
      Creates a 3 channel greyscale image and optionally
      sharpens it.
    """
    three_chan_grey = np.array(self.image)
    three_chan_grey[:, :, 0] = self.grey_image
    three_chan_grey[:, :, 1] = self.grey_image
    three_chan_grey[:, :, 2] = self.grey_image
    if self.apply_sharpening:
      sharp_grey = np.zeros(three_chan_grey.shape)
      sharp_grey[:, :, :] = three_chan_grey[:, : ,:]
      sharp_grey = 2*sharp_grey - ndimage.gaussian_filter(sharp_grey, 
                                                          self.sharpening_sigma)
      if self.normalize_grey:
        sharp_grey = sharp_grey - np.amin(sharp_grey)
        sharp_grey = 255*sharp_grey / np.amax(sharp_grey)
      three_chan_grey[:, :, :] =  sharp_grey[:, :, :]
    self.grey_mask = self.run_foward_pass(three_chan_grey)

  def get_color_image_mask(self):
    self.color_image_mask = self.run_foward_pass(self.image)
    self.color_image_only = self.color_image_mask

  def find_connected_components(self):
    """ 
      Identifies connected components and keeps those that
      intersect ENet predictions
    """
    if self.use_connected_components:
      blurred = cv2.medianBlur(self.grey_image ,5)
      _, binary_low = cv2.threshold(blurred, self.binary_lower_bound, 255, 
                                cv2.THRESH_BINARY)
      _, binary_high = cv2.threshold(blurred, self.binary_low_blue_bound, 255, 
                                cv2.THRESH_BINARY)
      binary_high[self.color_assertion_mask] = binary_low[self.color_assertion_mask]
      self.connected_components = label(binary_high == 255)
      self.connected_components[binary_high == 0] = 0
    else:
      self.connected_components = np.zeros((self.image.shape[0], 
                                            self.image.shape[1]), dtype=np.int)

  def get_color_assertion_mask(self):
    # The above doesn't work well from the ground but the following
    # simple rule seems effective for the connected components
    # assertion_1 = self.image[:, :, 0]  > self.image[:, :, 1]
    # assertion_2 = self.image[:, :, 0]  > self.image[:, :, 2]
    # self.color_assertion_mask = np.logical_and(assertion_1, assertion_2) 

    # IJCSI International Journal of Computer Science Issues, 
    # Vol. 10, Issue 4, No 1, July 2013
    # ISSN (Print): 1694-0814 | ISSN (Online): 1694-0784
    # www.IJCSI.org
    # if (abs(R - G)<5 && abs(G - B)<5 && B > R
    # && B>G && B>50 && B<230 ) 
    # output_4 = np.array(self.image)
    # rule_1 = np.abs(self.image[:, :, 2] - self.image[:, :, 1]) < 5
    # rule_2 = np.abs(self.image[:, :, 1] - self.image[:, :, 0]) < 5
    # rule_3 = self.image[: ,: ,0] > self.image[:, :, 2]
    # rule_4 = self.image[: ,: ,0] > self.image[:, :, 1]
    # rule_5 = self.image[: ,: ,0] > 50
    # rule_6 = self.image[: ,: ,0] < 230
    # all_checks = rule_1 * rule_2 * rule_3 * rule_4 * rule_5 * rule_6

    # use color to threshold binary cuttoff
    float_image = np.zeros(self.image.shape)
    float_image[:, :, :] = self.image[:, :, :]
    assertion_1 = float_image[:, :, 0]  > 1.3 * float_image[:, :, 1]
    assertion_2 = float_image[:, :, 0]  > 1.3 * float_image[:, :, 2]
    self.color_assertion_mask = np.logical_and(assertion_1, assertion_2) 

  def get_final_mask(self):
    intersection = np.array(self.connected_components)
    # Border cheat
    # self.color_image_mask[:20, :] = True
    intersection[np.logical_not(self.color_image_mask)] = 0
    remaining_labels = np.unique(intersection)
    self.remaining_components = np.logical_and(np.isin(self.connected_components, 
                                               remaining_labels), 
                                               self.connected_components !=0)
    self.final_mask = np.logical_or(self.color_image_mask, 
                                    self.remaining_components)