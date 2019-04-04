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

    def runVideo(self):
        if self.videoSrc == None:
            print("No video loaded!")
            return
        self.cap = cv2.VideoCapture()
        self.cap.open(self.videoSrc)

        while True:
            start_time = time.time()
            ret,img = self.cap.read()
            
            if not ret:
                self.cap.release()
                self.cap.open(self.videoSrc)
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
            small_sky_mask = np.zeros((360, 640), dtype=np.uint8)
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
    if source.endswith(('MOV', 'mp4')):
        ssl.videoSrc = source
        ssl.verbose = args.verbose
        if args.output is not None:
            ssl.videoOut = args.output
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
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  main()