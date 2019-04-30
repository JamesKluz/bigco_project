import argparse
import cv2
import numpy as np
import sys
import time

class SkySegmeterLight(object):
    def __init__(self): 
        self.cannyLow = 71
        self.cannyHigh = 208
        self.red_scalar = 1.15
        self.morphKernSize = (9,9)
        self.rp_threshVal = 200
        self.slp_binary_thresh = 170
        self.min_proportion_rp = 0.0025
        self.min_proportion_region = 0.01
        self.verbose = False
        self.videoSrc = None
        self.videoOut = None
        self.writer = None
        self.background = None
        self.background_list = []
        self.fade = False 
        self.rotate = False
        self.lock_left = False
        self.flip_vertical = False
        self.flip_horizontal = False
        self.high_speed = False
        self.hi_def = False
        self.break_cc_w_edges = False

    def demo(self):
        if self.videoSrc == None:
            print("No video loaded!")
            return

        self.cap = cv2.VideoCapture()
        self.cap.open(self.videoSrc)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("{} frames in video".format(total_frames))

        scaledSize = (640, 360)
        # if self.hi_def:
        #     scaledSize = (1240, 720)
        bg_images = []
        for image in self.background_list:
            bg_img = cv2.imread(image)
            if self.rotate:
                bg_img = np.rot90(bg_img, k=1)
            bg_img = cv2.resize(bg_img, scaledSize)
            bg_images.append(bg_img)

        bg_img = bg_images[0]
        frames_per_bg = 210
        switch_frames = 80
        image_index = -1
        count = -1
        first_pass = True

        while True:
            count = (count + 1) % frames_per_bg
            if count == 0:
                image_index = (image_index + 1) % len(bg_images)
                bg_img = bg_images[image_index]
            if count >= (frames_per_bg - switch_frames):
                mix_1 = float(frames_per_bg - count) / float(switch_frames)
                mix_2 = 1.0 - mix_1
                next_index = (image_index + 1) % len(bg_images)
                bg_img = mix_1 * bg_img + mix_2 * bg_images[next_index] 
            start_time = time.time()
            ret,img = self.cap.read()
            
            if not ret:
                self.cap.release()
                self.cap.open(self.videoSrc)
                continue

            if self.high_speed and count % 2 == 0:
                continue
            
            if img.shape[0] > img.shape[1]:
                scaledSize = (360, 640)
                if first_pass:
                    bg_img = np.rot90(bg_img, k=1)
                    first_pass = False
            image_size = float(scaledSize[0] * scaledSize[1])
            scaledImg = cv2.resize(img, scaledSize)

            grey = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            edges = cv2.Canny(grey, self.cannyLow, self.cannyHigh)
            blurred = cv2.medianBlur(grey ,5)
            sky_like_pixels = cv2.threshold(blurred, self.slp_binary_thresh, 255, cv2.THRESH_BINARY)[1]
            region_proposal = cv2.threshold(blurred, self.rp_threshVal, 255, cv2.THRESH_BINARY)[1]
            sky_whp = edges.copy()

            cv2.dilate(edges, 
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                       sky_whp, (-1,-1), 2)
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                             thresh, (-1,-1), 2)
            sky_whp = sky_whp | thresh
            sky_whp_temp = sky_whp.copy()
            sky_whp[sky_whp_temp == 0] = 255
            sky_whp[sky_whp_temp == 255] = 0
            sky_whp[sky_like_pixels == 255] = 255
            sky_whp[self.red_scalar * scaledImg[:, :, 0] < scaledImg[:, :, 2]] = 0
            if self.break_cc_w_edges:
                sky_whp[edges == 255] = 0

            ############################## CC ##############################
            ret, labels = cv2.connectedComponents(sky_whp)
            region_size = np.bincount(labels.reshape((labels.shape[0] * labels.shape[1], )))
            label_hue = np.uint8(179*labels/np.max(labels))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[label_hue==0] = 0
            ############################## CC ##############################
            ############################## CC ##############################
            _, rp_labels = cv2.connectedComponents(region_proposal)
            rp_size = np.bincount(rp_labels.reshape((rp_labels.shape[0] * rp_labels.shape[1], )))
            rp_components = np.unique(rp_labels)
            if rp_components.size > 0 and rp_components[0] == 0:
                rp_components = rp_components[1:]
            good_components = []
            for c in rp_components:
                pixel_count = rp_size[c]
                if pixel_count > (self.min_proportion_rp * image_size):
                    good_components.append(c)
            rp_components = np.array(good_components)
            region_proposal[np.logical_not(np.isin(rp_labels, rp_components))] = 0
            ############################## CC ##############################
            ########################### MERGE CC ###########################
            sky_components = np.unique(labels[region_proposal == 255])
            if sky_components.size > 0 and sky_components[0] == 0:
                sky_components = sky_components[1:]
            good_components = []
            for c in sky_components:
                pixel_count = region_size[c]
                if pixel_count > (self.min_proportion_region * image_size):
                    good_components.append(c)
            sky_components = np.array(good_components) 
            # print(sky_components)
            ########################### MERGE CC ###########################
            #make masked images
            small_sky_mask = np.zeros((scaledSize[1], scaledSize[0]), dtype=np.uint8)
            small_sky_mask[np.isin(labels, sky_components)] = 1
            # THIS IS FOR CREATING FINAL OUTPUT
            output = np.zeros((2*scaledImg.shape[0] + 20, scaledImg.shape[1], 
                               scaledImg.shape[2]), dtype=np.uint8)
            final_image = scaledImg.copy()
            # final_image[small_sky_mask == 1] = np.array([255, 0, 0], dtype=np.uint8)
            final_image[small_sky_mask == 1] = bg_img[small_sky_mask == 1]
            output[:scaledImg.shape[0], :] = scaledImg
            output[scaledImg.shape[0] + 20:, :] = final_image

            if self.hi_def:
                output = cv2.resize(output, (int(1.35 * scaledSize[0]), int(2.7 * scaledSize[1])))

            # write images if output provided:
            if self.videoOut is not None and self.writer is None:
                # initialize our video writer
                vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                self.writer = cv2.VideoWriter(self.videoOut, vw_fourcc, 30,
                                             (output.shape[1], output.shape[0]), 
                                              True)

            #show images
            winname = "The Mighty Four Demo!"
            cv2.namedWindow(winname)    
            if self.lock_left:
                cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
            cv2.imshow(winname, output)
            end_time = time.time()
            fps = (1.0 / (end_time - start_time))
            if self.verbose:
                print("{} fps".format(fps))

            ch = 0xFF & cv2.waitKey(5)
            if ch == ord('q'):
                break

            # write the output frame to disk
            if self.videoOut:
                self.writer.write(output)

        self.writer.release()
        self.writer = None

    def addBackground(self):
        if self.videoSrc == None:
            print("No video loaded!")
            return

        bg_img = cv2.imread(self.background)
        if self.rotate:
            bg_img = img = np.rot90(bg_img, k=1)
        print(bg_img.shape)
        self.cap = cv2.VideoCapture()
        self.cap.open(self.videoSrc)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("{} frames in video".format(total_frames))

        scaledSize = (640, 360)
        # if self.hi_def:
        #     scaledSize = (1240, 720)
        bg_img = cv2.resize(bg_img, scaledSize)

        mix = 0

        first_pass = True

        while True:
            mix = (mix + 1) % total_frames
            start_time = time.time()
            ret,img = self.cap.read()
            
            if not ret:
                self.cap.release()
                self.cap.open(self.videoSrc)
                continue

            if self.high_speed and mix % 2 == 0:
                continue
            
            if img.shape[0] > img.shape[1]:
                scaledSize = (360, 640)
                if first_pass:
                    bg_img = np.rot90(bg_img, k=1)
                    first_pass = False
            image_size = float(scaledSize[0] * scaledSize[1])
            scaledImg = cv2.resize(img, scaledSize)

            grey = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            edges = cv2.Canny(grey, self.cannyLow, self.cannyHigh)
            blurred = cv2.medianBlur(grey ,5)
            sky_like_pixels = cv2.threshold(blurred, self.slp_binary_thresh, 255, cv2.THRESH_BINARY)[1]
            region_proposal = cv2.threshold(blurred, self.rp_threshVal, 255, cv2.THRESH_BINARY)[1]
            sky_whp = edges.copy()

            cv2.dilate(edges, 
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                       sky_whp, (-1,-1), 2)
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                             thresh, (-1,-1), 2)
            sky_whp = sky_whp | thresh
            sky_whp_temp = sky_whp.copy()
            sky_whp[sky_whp_temp == 0] = 255
            sky_whp[sky_whp_temp == 255] = 0
            sky_whp[sky_like_pixels == 255] = 255
            sky_whp[self.red_scalar * scaledImg[:, :, 0] < scaledImg[:, :, 2]] = 0
            if self.break_cc_w_edges:
                sky_whp[edges == 255] = 0

            ############################## CC ##############################
            ret, labels = cv2.connectedComponents(sky_whp)
            region_size = np.bincount(labels.reshape((labels.shape[0] * labels.shape[1], )))
            label_hue = np.uint8(179*labels/np.max(labels))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[label_hue==0] = 0
            ############################## CC ##############################
            ############################## CC ##############################
            _, rp_labels = cv2.connectedComponents(region_proposal)
            rp_size = np.bincount(rp_labels.reshape((rp_labels.shape[0] * rp_labels.shape[1], )))
            rp_components = np.unique(rp_labels)
            if rp_components.size > 0 and rp_components[0] == 0:
                rp_components = rp_components[1:]
            good_components = []
            for c in rp_components:
                pixel_count = rp_size[c]
                if pixel_count > (self.min_proportion_rp * image_size):
                    good_components.append(c)
            rp_components = np.array(good_components)
            region_proposal[np.logical_not(np.isin(rp_labels, rp_components))] = 0
            ############################## CC ##############################
            ########################### MERGE CC ###########################
            sky_components = np.unique(labels[region_proposal == 255])
            if sky_components.size > 0 and sky_components[0] == 0:
                sky_components = sky_components[1:]
            good_components = []
            for c in sky_components:
                pixel_count = region_size[c]
                if pixel_count > (self.min_proportion_region * image_size):
                    good_components.append(c)
            sky_components = np.array(good_components) 
            # print(sky_components)
            ########################### MERGE CC ###########################
            #make masked images
            small_sky_mask = np.zeros((scaledSize[1], scaledSize[0]), dtype=np.uint8)
            small_sky_mask[np.isin(labels, sky_components)] = 1
            # THIS IS FOR CREATING SMALLER FINAL OUTPUT
            final_image = scaledImg.copy()
            # final_image[small_sky_mask == 1] = np.array([255, 0, 0], dtype=np.uint8)
            if self.videoOut is not None and mix == (total_frames - 1):
                break
            if self.fade:
                new_mix = float(mix) / float(total_frames)
                if new_mix < 0.10:
                    new_mix = 0.0
                elif new_mix > 0.20 and new_mix < .80:
                    new_mix = 1.0
                elif new_mix >= 0.1 and new_mix <= .20:
                    new_mix = 10*(new_mix - 0.10)
                elif new_mix < 0.9:
                    new_mix = 1.0 - 10*(new_mix - 0.80)
                else:
                    new_mix = 0.0

                og_mix = 1.0 - new_mix

                
                final_image[small_sky_mask == 1] = new_mix * bg_img[small_sky_mask == 1] + og_mix * final_image[small_sky_mask == 1]
            else:
                final_image[small_sky_mask == 1] = bg_img[small_sky_mask == 1]

            if self.hi_def:
                final_image = cv2.resize(final_image, (2 * scaledSize[0], 2 * scaledSize[1]))

            # write images if output provided:
            if self.videoOut is not None and self.writer is None:
                # initialize our video writer
                vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                self.writer = cv2.VideoWriter(self.videoOut, vw_fourcc, 30,
                                             (final_image.shape[1], final_image.shape[0]), 
                                              True)

            #show images
            cv2.imshow("Visualization", final_image)
            end_time = time.time()
            fps = (1.0 / (end_time - start_time))
            if self.verbose:
                print("{} fps".format(fps))

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break

            # write the output frame to disk
            if self.videoOut:
                self.writer.write(final_image)
        self.writer.release()
        self.writer = None

    def runVideo(self):
        if self.videoSrc == None:
            print("No video loaded!")
            return
        self.cap = cv2.VideoCapture()
        self.cap.open(self.videoSrc)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("{} frames in video".format(total_frames))

        count = 0

        while True:
            count += 1
            if self.videoOut is not None and count == 4*total_frames:
                break
            start_time = time.time()
            ret,img = self.cap.read()
            
            if not ret:
                self.cap.release()
                self.cap.open(self.videoSrc)
                continue

            if self.high_speed and count % 2 == 0:
                continue
            
            scaledSize = (640, 360)
            if img.shape[0] > img.shape[1]:
                scaledSize = (360, 640)
            image_size = float(scaledSize[0] * scaledSize[1])
            scaledImg = cv2.resize(img, scaledSize)

            grey = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            edges = cv2.Canny(grey, self.cannyLow, self.cannyHigh)
            blurred = cv2.medianBlur(grey ,5)
            sky_like_pixels = cv2.threshold(blurred, self.slp_binary_thresh, 255, cv2.THRESH_BINARY)[1]
            region_proposal = cv2.threshold(blurred, self.rp_threshVal, 255, cv2.THRESH_BINARY)[1]
            sky_whp = edges.copy()

            cv2.dilate(edges, 
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                       sky_whp, (-1,-1), 2)
            cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                             thresh, (-1,-1), 2)
            sky_whp = sky_whp | thresh
            sky_whp_temp = sky_whp.copy()
            sky_whp[sky_whp_temp == 0] = 255
            sky_whp[sky_whp_temp == 255] = 0
            sky_whp[sky_like_pixels == 255] = 255
            sky_whp[self.red_scalar * scaledImg[:, :, 0] < scaledImg[:, :, 2]] = 0
            if self.break_cc_w_edges:
                sky_whp[edges == 255] = 0

            ############################## CC ##############################
            ret, labels = cv2.connectedComponents(sky_whp)
            region_size = np.bincount(labels.reshape((labels.shape[0] * labels.shape[1], )))
            label_hue = np.uint8(179*labels/np.max(labels))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            labeled_img[label_hue==0] = 0
            ############################## CC ##############################
            ############################## CC ##############################
            _, rp_labels = cv2.connectedComponents(region_proposal)
            rp_size = np.bincount(rp_labels.reshape((rp_labels.shape[0] * rp_labels.shape[1], )))
            rp_components = np.unique(rp_labels)
            if rp_components.size > 0 and rp_components[0] == 0:
                rp_components = rp_components[1:]
            good_components = []
            for c in rp_components:
                pixel_count = rp_size[c]
                if pixel_count > (self.min_proportion_rp * image_size):
                    good_components.append(c)
            rp_components = np.array(good_components)
            region_proposal[np.logical_not(np.isin(rp_labels, rp_components))] = 0
            ############################## CC ##############################
            ########################### MERGE CC ###########################
            sky_components = np.unique(labels[region_proposal == 255])
            if sky_components.size > 0 and sky_components[0] == 0:
                sky_components = sky_components[1:]
            good_components = []
            for c in sky_components:
                pixel_count = region_size[c]
                if pixel_count > (self.min_proportion_region * image_size):
                    good_components.append(c)
            sky_components = np.array(good_components) 
            # print(sky_components)
            ########################### MERGE CC ###########################
            #make masked images
            small_sky_mask = np.zeros((scaledSize[1], scaledSize[0]), dtype=np.uint8)
            small_sky_mask[np.isin(labels, sky_components)] = 1
            # THIS IS FOR CREATING SMALLER FINAL OUTPUT
            final_image = scaledImg.copy()
            final_image[small_sky_mask == 1] = np.array([255, 0, 0], dtype=np.uint8)

            height = final_image.shape[0]
            width = final_image.shape[1]
            border = 10
            output = np.zeros((2 * height + 3*border, 2 * width + 3*border, 3), dtype=np.uint8)
            output += 255
            output[border:height + border, border:width + border] = scaledImg
            output[height + 2*border : -border, border : width + border, 0] = edges
            output[height + 2*border : -border, border : width + border, 1] = edges
            output[height + 2*border : -border, border : width + border, 2] = edges
            output[border : height + border, width + 2*border: -border] = labeled_img
            output[height + 2*border: - border, width + 2*border: -border] = final_image

            # write images if output provided:
            if self.videoOut is not None and self.writer is None:
                # initialize our video writer
                vw_fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                self.writer = cv2.VideoWriter(self.videoOut, vw_fourcc, 30,
                                             (output.shape[1], output.shape[0]), 
                                              True)

            #show images
            cv2.imshow("Visualization", output)
            end_time = time.time()
            fps = (1.0 / (end_time - start_time))
            if self.verbose:
                print("{} fps".format(fps))

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break

            # write the output frame to disk
            if self.videoOut:
                self.writer.write(output)
        self.writer.release()
        self.writer = None


    def getSkyMask(self, img):        
        scaledSize = (640, 360)
        if img.shape[0] > img.shape[1]:
            scaledSize = (360, 640)
        image_size = float(scaledSize[0] * scaledSize[1])
        scaledImg = cv2.resize(img, scaledSize)

        grey = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        edges = cv2.Canny(grey, self.cannyLow, self.cannyHigh)
        blurred = cv2.medianBlur(grey ,5)
        sky_like_pixels = cv2.threshold(blurred, self.slp_binary_thresh, 255, cv2.THRESH_BINARY)[1]
        region_proposal = cv2.threshold(blurred, self.rp_threshVal, 255, cv2.THRESH_BINARY)[1]
        sky_whp = edges.copy()

        cv2.dilate(edges, 
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                   sky_whp, (-1,-1), 2)
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                         thresh, (-1,-1), 2)
        sky_whp = sky_whp | thresh
        sky_whp_temp = sky_whp.copy()
        sky_whp[sky_whp_temp == 0] = 255
        sky_whp[sky_whp_temp == 255] = 0
        sky_whp[sky_like_pixels == 255] = 255
        sky_whp[self.red_scalar * scaledImg[:, :, 0] < scaledImg[:, :, 2]] = 0

        ############################## CC ##############################
        ret, labels = cv2.connectedComponents(sky_whp)
        region_size = np.bincount(labels.reshape((labels.shape[0] * labels.shape[1], )))
        ############################## CC ##############################
        ############################## CC ##############################
        _, rp_labels = cv2.connectedComponents(region_proposal)
        rp_size = np.bincount(rp_labels.reshape((rp_labels.shape[0] * rp_labels.shape[1], )))
        rp_components = np.unique(rp_labels)
        if rp_components.size > 0 and rp_components[0] == 0:
            rp_components = rp_components[1:]
        good_components = []
        for c in rp_components:
            pixel_count = rp_size[c]
            if pixel_count > (self.min_proportion_rp * image_size):
                good_components.append(c)
        rp_components = np.array(good_components)
        region_proposal[np.logical_not(np.isin(rp_labels, rp_components))] = 0
        ############################## CC ##############################
        ########################### MERGE CC ###########################
        sky_components = np.unique(labels[region_proposal == 255])
        if sky_components.size > 0 and sky_components[0] == 0:
            sky_components = sky_components[1:]
        good_components = []
        for c in sky_components:
            pixel_count = region_size[c]
            if pixel_count > (self.min_proportion_region * image_size):
                good_components.append(c)
        sky_components = np.array(good_components) 
        ########################### MERGE CC ###########################
        #make masked images
        small_sky_mask = np.zeros((scaledSize[1], scaledSize[0]), dtype=np.uint8)
        small_sky_mask[np.isin(labels, sky_components)] = 1
        full_sky_mask = cv2.resize(small_sky_mask, 
                                   (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return full_sky_mask

def main():
    source = args.input
    ssl = SkySegmeterLight()
    if args.background is not None:
        ssl.background = args.background
    if source.endswith(('MOV', 'mp4', 'mov')):
        ssl.videoSrc = source
        ssl.verbose = args.verbose
        ssl.fade = args.fade
        ssl.min_proportion_region = args.rp_pp
        ssl.high_speed = args.hs
        ssl.hi_def = args.hd
        ssl.break_cc_w_edges = args.break_cc_w_edges
        if args.output is not None:
            ssl.videoOut = args.output
        if args.background is not None:
            ssl.addBackground()
        else:
            ssl.runVideo()
    else:
        img = cv2.imread(source) 
        mask = ssl.getSkyMask(img)
        h, w = (960, 540)
        if img.shape[0] > img.shape[1]:
            h, w = (540, 960)
        masked_image = img.copy()
        masked_image[mask==1] = np.array([255, 0, 0])
        print(masked_image.shape)
        if args.output is not None:
            cv2.imwrite(args.output, masked_image)
        img  = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
        masked_image  = cv2.resize(masked_image, 
                                  (h, w), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Input", img)
        cv2.imshow("Output", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def parse_args():
  parser = argparse.ArgumentParser(
      description="Sky segmentation script for images and videos"
  )
  parser.add_argument("input", help="Path to input file")
  parser.add_argument("--output", help="Path to output file (optional)")
  parser.add_argument("--verbose", help="output info like fps", action="store_true")
  parser.add_argument("--background", help="path to background to paste on image/video")
  parser.add_argument("--fade", help="fade from real to background", action="store_true")
  parser.add_argument("--hs", help="drop frames to stay in real time", action="store_true")
  parser.add_argument("--hd", help="output hi res video", action="store_true")
  parser.add_argument("--break_cc_w_edges", help="break connected components with edges", action="store_true")
  parser.add_argument("--rp_pp", type=float, default=0.01, help="region proposal proportion")
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main()