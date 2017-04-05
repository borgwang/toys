Use a neural net to simply mimic a image.  
The inpu of the net is the coordinate of pixels of the image i.e. (x,y) and the ground true output is the corresponding RGB of that pixel.

## Example  

<img src="https://github.com/borgwang/toys/raw/master/nn_paint/res/origin.jpg" width = "256" height = "160" alt="origin" align=center />   
Origin Image  
<img src="https://github.com/borgwang/toys/raw/master/nn_paint/res/paint.jpg" width = "256" height = "160" alt="paint" align=center />   
Paint Image  

## Requirements
* python 2.7  
* Tensorflow  
* Numpy  
* Pillow  

## Usage  
1. Put an origin.jpg piture in **./res/**   
2. Run  main.py  
3. The output will be stored at **./res/paint.jpg**  


## Architecture  
We form a 7-layer full-connected neural net and train it with a Momentum optimizer(learning rate=0.01, momentum=0.9).      
Feel free to modify the code to build your own 'painter'.  

