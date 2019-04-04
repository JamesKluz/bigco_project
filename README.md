# Sky Segmentation

## Authors: James Kluz

### Description:
- Sky Segmenter is an algorithm that takes as input images or videos of outdoor scenes and returns a per-pixel binary mask representing `sky/not-sky`. The algorithm utilizes the ENET network developed by Adam Paszke et al, as well as various clustering, connected components and color-based classification approaches. The algorithm was trained and tuned utilizing the CityScapes dataset developed by Marius Cordts et al.

- Sky Segmenter currently runs at about 3 FPS on a CPU. Since we hope for this model to run on a cell phone in tandem with additional 3D processing we need something with much less computational complexity. Do to the large amount of parameters in even the smallest of deep learning models capable of per-pixel segmentation we are currently working on a probability based approach using GMMs to create our sky segmentation masks which we are calling SkySegmenterLight. This new model runs at about 50 fps. Instructions for using this model are at the bottom of this README.

- This project is part of a larger effort to design an augmeneted reality application for AccuWeather as part of the BigCo curriculum at Cornell Tech.  

## Requirements:
- python 2 (SkySegmenterLight runs in both python 2/3)
- numpy
- scikit-learn
- opencv
- scipy
- imutils

### To Do:
- Implement GMM for close calls
- Use edges to identify 'smooth' connected components

### Ideas:
- Feature points for locking mask to scene between CNN returns
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