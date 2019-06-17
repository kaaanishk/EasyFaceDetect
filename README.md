# Optimised Face Detection

## Introduction
This repository uses an implementation of MTCNN in Tensorflow for Python 3.4+. The implementation is based off [FaceNet's implementation of MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align) which is based on the paper Zhang, K et al. (2016) <sup>[1](#references)</sup>

All procedures done are to optimise performance of the MTCNN implementation on a private dataset.
I decided to test out the following methods to improve detection results.
- Gamma Correction
- Contrast Limited Adaptive Histogram Equalisation (CLAHE)
- Tan and Triggs Illumination Normalisation <sup> [2](#references)</sup>
- Image Resizing

Through trial and testing (which can be seen in the Google Colab Notebook), the following decisions were taken:
- Removal of Tan and Triggs Illumination Normalisation from the Face Detection Pipeline since it showed decreased performance on the given dataset
- The best value of gamma to improve dynamic range turns out to be **1.6**

>**Note:** Tan and Triggs Illumination Normalisation will come in handy while extracting features from cropped faces to do Face Recognition, therefore the code is preserved in the repository.

## Progress
- [x] Implement and fine-tune basic image pre-processing techniques
- [x] Implement Tan and Triggs Illumination Normalisation according to Paper
- [x] Implement MTCNN Face Detection
- [ ] Crop Faces to `112x112` to pass onto Face Recognition Module


## References

|Link          |Name                                                                 |
|:--------------|:---------------------------------------------------------------------|
| [\[ZHANG2016\]](https://arxiv.org/abs/1604.02878v1) | Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.|
| [\[TANTRIGGS2010\]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5411802&isnumber=5464460) | X. Tan and B. Triggs, "Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions," in IEEE Transactions on Image Processing, vol. 19, no. 6, pp. 1635-1650, June 2010.|
