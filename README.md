# 3D object detection problem


This is my project as a part of internship in ELEKS. My task was to predict 3D bounding boxes for objects on photos. 
To this end, I chose Google [Objectron](https://github.com/google-research-datasets/Objectron) dataset as my primary data.

***
At first, I tried VGG16 model with frozen convolutional layers. After 30 epochs we see it starts overfitting at 7-th epoch.
![VGG16_froze_weights](https://user-images.githubusercontent.com/76902422/191322750-2e1385f4-e2dd-4b74-819d-e122495ddb6c.png)

Then I trained it for 7 epochs, unfroze weights and trained it again for 10 epochs.
