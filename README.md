# Neural-Network-with-CUDA
The project is a follow up on my Nueral network in c/cpp
This project implements several mathematical functions in CUDA. It also applies the ADAM optimizer.

Some interesting findings:
CUDA did not speed up the neural netowrk. This is likely due to the fact that the arrays and data from the mnist data set wern't large enought to make up for the time the computer took to set up a gpu kernal.
The ADAM optimizer helped significantly. It trained my data in the first epoch and had a higher accuracy rate than any other tests before. However after the 3rd epoch the accuracy went down a little bit. This makes me think that it started to memorize the patterns and overfit instead of learning them.

I hope to return to this project in the future and implement some other nueral netork components and theories.

I completed this is visual studio and was unable to get the proper files to push so I uploaded the source files to github.
