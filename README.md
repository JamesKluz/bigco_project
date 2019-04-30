# Sky Segmentation

## Authors: James Kluz

### Demo Quick Start Guide:
Below are the steps to run the demo. The demo will process the video demo.mp4 in the `videos` directory on the fly, periodically replacing the sky with different weather patterns. A before/after window will be displayed with the original video on the top and the processed video on the bottom. Alternatively, the demo_video.mp4 file in this repo is recorded output of running the demo.
- install numpy
- install opencv (opencv-python)
- Run the following: `python3 demo.py --lock_left`
- The player will appear on the left side of the screen so make sure it's clear. If the above puts the viewer off screen try: `python3 demo.py`
- To run the demo with your own video run: `python3 demo.py --input <path to video>`
- feel free to contact me with any questions: jwk259@cornell.edu

### Description:
- Sky Segmenter is an algorithm that takes as input images or videos of outdoor scenes and returns a per-pixel binary mask representing `sky/not-sky`. The algorithm utilizes the ENET network developed by Adam Paszke et al, as well as various clustering, connected components and color-based classification approaches. The algorithm was trained and tuned utilizing the CityScapes dataset developed by Marius Cordts et al.

- The original Sky Segmenter runs at about 3 FPS on a CPU. Since we hope for this model to run on a cell phone in tandem with additional 3D processing we needed something with much less computational complexity. Do to the large amount of parameters in even the smallest of deep learning models capable of per-pixel segmentation we traded in the neural net for a probability distribution based approach which resulted in a different system that we are referring to as SkySegmenterLight. This new model runs at about 50 fps. Instructions for using both models are at the bottom of this README.

- This project is part of a larger effort to design an augmeneted reality application for AccuWeather as part of the BigCo curriculum at Cornell Tech.  

## Requirements SkySegmenterLight:
- python 2 or 3 
- numpy
- opencv 

## Requirements SkySegmenter:
- python 2
- numpy
- scikit-learn
- opencv
- scipy
- imutils

### Ideas:
- Probability map to carry predictions from one frame to the next
- Use knowledge of location and current weather conditions to call models fine-tuned for those specific lighting conditions

## Execution SkySegmenter:
### Segment Images:
- Run the following:
`python render_sky_mask.py <path to image> --output <optional path to image output> --no_normalize`

#### Where:
* `<path to image>` is the path to the image to apply the mask
* `--output <optional path to image output>` optional path to write masked image
* `--no_normalize` optional but currently reccomended. More testing and hyper-parameter tuning is needed. 

### Segment Video:
`python render_sky_mask.py <path to video input> --output <path to video output> --video --no_normalize`

#### Where:
* `<path to video input>` is the path to the video to apply the mask
* `<path to video output>` path to write masked video output. Should be .avi extension
* `--no_normalize` optional but currently reccomended. More testing and hyper-parameter tuning is needed. 

## Execution SkySegmenterLight:
### Segment Images:
- Run the following:
`python SkySegmenterlight.py <path to image> --output <optional path to image output>`

#### Where:
* `<path to image>` is the path to the image to apply the mask
* `--output <optional path to image output>` optional path to write masked image

### Segment Video:
- Run the following:
`python SkySegmenterlight.py <path to video> --output <optional path to video output>`

#### Where:
* `<path to video>` is the path to the video to apply the mask
* `--output <optional path to video output>` optional path to write masked video