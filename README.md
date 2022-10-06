# 3D object detection problem


This is my project as a part of internship in ELEKS. My task was to predict 3D bounding boxes for objects on photos. 
To this end, I chose Google [Objectron](https://github.com/google-research-datasets/Objectron) dataset as my primary data.

***
At first, I tried VGG16 model with frozen convolutional layers. After 30 epochs we see it starts overfitting at 7-th epoch.

![VGG16_froze_weights](https://user-images.githubusercontent.com/76902422/191322750-2e1385f4-e2dd-4b74-819d-e122495ddb6c.png)

Then I unfroze weights and trained the model again. 

![VGG16_unfroze_weights_15epochs](https://user-images.githubusercontent.com/76902422/194294088-b6561456-5724-484c-93a4-78abcdae7cb0.png)

Unfortunatelly, it didn't help a bit.
