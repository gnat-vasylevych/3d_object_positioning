# 3D object detection problem


This is my project as a part of internship in ELEKS. My task was to predict 3D bounding boxes for objects on photos. 
To this end, I chose Google [Objectron](https://github.com/google-research-datasets/Objectron) dataset as my primary data.
There are several classes available. For simplicity I chose only one - cup class. Also, I train my model only on photos with one cup in it, omitting pictures with two or more cups.

***
Google's annotation provides coordinates of bounding box for each frame. It describes 9 points id 3d coordinate system (8 of a box and 1 in the center of a box), so 27 values total.

At first I build a simple custom model which predicts 27 values to get baseline.

![first_custom_model](https://user-images.githubusercontent.com/76902422/195856121-5c2c8dac-105f-4944-84b7-b8c0201dacf4.png)

This model produced MSE loss value at about 0.1805
![image](https://user-images.githubusercontent.com/76902422/195857795-37573da7-8644-4a51-a6b9-6d842035661d.png)

Then I tried VGG16 model with frozen convolutional layers.

![VGG16_froze_weights](https://user-images.githubusercontent.com/76902422/191322750-2e1385f4-e2dd-4b74-819d-e122495ddb6c.png)

The result is much better than baseline - the MSE is 0.022

Then I unfroze weights and trained the model again. 

![VGG16_unfroze_weights_15epochs](https://user-images.githubusercontent.com/76902422/194294088-b6561456-5724-484c-93a4-78abcdae7cb0.png)

Unfortunatelly, it didn't help a bit.
