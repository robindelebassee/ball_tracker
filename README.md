# ball_tracker
Tracker tool to follow the ball in volleyball video tapes


Link to Google Drive data repo for the project 
https://drive.google.com/drive/folders/1EhpmRTL258UyIT3EJpO_CIbpX_SZIpX9?usp=sharing


Possible approaches for ball-tracking : 

Color based approach : 
    Ball is either green/white/red or blue/yellow. Could be simple enough and work fine on some video.
    Problem for generalization, if background is same color than the ball or if ball color is new.
    If the ball move too quickly the colors fade on the frames and make detection more difficult.
    
Hough circles approach : 
    Detect ball using Hough circles. Search to explain how it works.
    First results seem limited because the ball quality on the frame is too low.
    
Background/Forground extraction : 
    Extract from image the moving objects, among which the ball should be.
    Look for the highest node detected, there are good chances it's the ball.
    
Super Pixel : 
    Transforming frames using superpixel to see if it allows for a better ball detection. 
    (i.e. if it can be detected as a cluster of superpixels)

K-means :
    Can be used after superpixel.

Mean Shift : 
    Used to be state of the art for a while.
    Can be used after superpixel ?

Region based approach : 
    To keep spatial consistance (and allow to process the ball as one entity ?)

Graph based approach :
    Process pixels as nodes of a graph, edges are similarity relations between concurrent pixels. 
    Regionalization computed through minimal cost graph cut --> region detection.
    
           