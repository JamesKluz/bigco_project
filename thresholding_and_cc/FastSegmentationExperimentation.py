import cv2
import numpy as np
import sys
import time

class SkySegmeterLight(object):
    def __init__(self, video_src):
        self.cannyLow = 71
        self.cannyHigh = 208
        self.red_scalar = 1.15
        self.morphKernSize = (9,9)
        self.rp_threshVal = 200
        self.slp_binary_thresh = 170
        self.min_proportion_rp = 0.0025
        self.min_proportion_region = 0.01
        self.verbose = False

        self.videoSrc = video_src

        self.cap = cv2.VideoCapture()
        self.cap.open(self.videoSrc)
        
        #window names
        self.inputWinName = "input"
        self.edgeWinName = "edge"
        self.outputWinName = "final"

        cv2.namedWindow(self.inputWinName)
        cv2.namedWindow(self.outputWinName)
        cv2.namedWindow(self.edgeWinName)

        cv2.createTrackbar("rp_bin_thresh", self.outputWinName, self.rp_threshVal, 1000, self.onSliderRPBT)
        cv2.createTrackbar("Region Size", self.outputWinName, 10, 1000, self.onSliderRegionSize)
        cv2.createTrackbar("red scalar", self.outputWinName, 15, 100, self.onSliderRS)

    def onSliderRPBT(self, val):
        self.rp_threshVal = val

    def onSliderRegionSize(self, val):
        self.min_proportion_region = float(val) / 1000.0

    def onSliderRS(self, val):
        self.red_scalar = 1.0 + float(val) / 100.0

    def runVideo(self):

        scaleFactor = 0.5

        while True:
            start_time = time.time()
            # next frame
            ret,img = self.cap.read()
            
            if not ret:
                self.cap.release()
                self.cap.open(self.videoSrc)
                continue
            
            scaledSize = (640, 360)
            image_size = float(scaledSize[0] * scaledSize[1])
            scaledImg = cv2.resize(img, scaledSize)

            grey = cv2.cvtColor(scaledImg, cv2.COLOR_BGR2GRAY)
    
            #equalize histogram
            #grey = cv2.equalizeHist(grey)

            ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            edges = cv2.Canny(grey, self.cannyLow, self.cannyHigh)

            blurred = cv2.medianBlur(grey ,5)
            sky_like_pixels = cv2.threshold(blurred, self.slp_binary_thresh, 255, cv2.THRESH_BINARY)[1]
            
            region_proposal = cv2.threshold(blurred, self.rp_threshVal, 255, cv2.THRESH_BINARY)[1]

            workImg = edges.copy()

            cv2.dilate(         edges,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                                workImg,
                                (-1,-1),
                                2)
            cv2.morphologyEx(   thresh,
                                cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,self.morphKernSize),
                                thresh,
                                (-1,-1),
                                2)

            
            workImg |= thresh

            workImg_temp = workImg.copy()
            workImg[workImg_temp == 0] = 255
            workImg[workImg_temp == 255] = 0
            workImg[sky_like_pixels == 255] = 255
            workImg[self.red_scalar * scaledImg[:, :, 0] < scaledImg[:, :, 2]] = 0

            ############################## CC ##############################
            ret, labels = cv2.connectedComponents(workImg)
            region_size = np.bincount(labels.reshape((labels.shape[0] * labels.shape[1], )))
            # print(labels.shape)
            # print(np.unique(labels))
            # Map component labels to hue val
            label_hue = np.uint8(179*labels/np.max(labels))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
            # cvt to BGR for display
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
            # set bg label to black
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
            # THIS IS FOR CREATING FULL MASK
            # full_sky_mask = cv2.resize(small_sky_mask, 
            #                            (img.shape[1], img.shape[0]),
            #                            interpolation=cv2.INTER_NEAREST)
            # img[full_sky_mask == 1] = np.array([255, 0, 0], dtype=np.uint8)
            # final_image = img

            #show images
            cv2.imshow(self.inputWinName, scaledImg)
            # cv2.imshow(self.edgeWinName, thresh)
            cv2.imshow(self.edgeWinName, edges)
            cv2.imshow("Labeled", labeled_img)
            cv2.imshow(self.outputWinName, final_image)

            end_time = time.time()
            fps = (1.0 / (end_time - start_time))
            if self.verbose:
                print("{} fps".format(fps))

            

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
    
        self.cap.release()

if __name__ == '__main__':
    try: 
        source = sys.argv[1]
    except: 
        sys.exit(1)
    if source.endswith(('MOV', 'mp4')):
        ssl = SkySegmeterLight(source)
        ssl.verbose = True
        ssl.runVideo()