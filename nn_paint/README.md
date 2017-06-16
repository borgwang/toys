Use a Neural Net to simply mimic a image.  
The input to the Net is the coordinate of pixels of the image i.e. (x,y) and the ground true output is the corresponding RGB of that pixel.
Basically it's a simple regression problem which the Neural Net tries to figure out the relation between coordinate and RGB value. It implements some kind of 'smooth' effect.

## Example  
Origin Image  
<img src="https://github.com/borgwang/toys/raw/master/nn_paint/res/origin.jpg" width = "256" height = "160" alt="origin" align=center />  

Paint Image  
<img src="https://github.com/borgwang/toys/raw/master/nn_paint/res/paint.jpg" width = "256" height = "160" alt="paint" align=center />   


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
A 7-layer full-connected neural net was formed and trained with a Momentum optimizer(lr=0.01, momentum=0.9).      
Feel free to modify the code to build your own painter.  
