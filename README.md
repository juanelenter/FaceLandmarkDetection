# FaceLandmarkDetection

## Early approaches: Fitting a deformable face mesh using strong priors.
- Active Shape Model
- Active Appearance Model
- Constrained Local Model

Only achieve good results in controlled environments (frontal face and proper lightning).

## Established methods.
- ERT: cascade based on gradient boosting. :heavy_check_mark:

Paper: One Millisecond Face Alignment with an Ensemble of Regression Trees paper by Kazemi and Sullivan (2014).

In order to run ERT you should download the pretrained model from: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ in the "--shape-predictor" section and then extract the ".dat" file to the main folder.

![landmarks](https://github.com/juanelenter/FaceLandmarkDetection/images/68landmarks.PNG)

- MultiTask Cascaded CNN: 

## Modern developments
- Dense Face Alignment
- Style Aggregated Network
- Look at Boundary  
- Wing Loss
- PFLD
- AWing
- Geometry Aggregated Network
- Deep Adaptive Graph
