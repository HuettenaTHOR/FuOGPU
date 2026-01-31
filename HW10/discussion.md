# Homework 8 - Discussion
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

# Task Configuration
To match the homework requirements, valid hyperparameters had to be found beforehand, so host-to-device, kernel execution and device-to-host all take approximately the same time. For this a target time of 8 ms was set. Different quadradic image sizes were tested until a matching size (3584 x 3584) was found which had an execution time > 8ms.  
Same was done for the mock-kernel which simply copies input to the output. Here, the number of repetitions could be changed to increase execution time to also match the required 8 ms. Each run results in a slightly different number of repetetitions but all lie in a range of 120-140 for the used GPU (RTX 4060 ti).  
The config for the upcoming tests therefore was a square image of size 3584 x 3584 and a number of repetitions for the kernel of 132.

# Task 1: 100 images processed without streaming

# Task 2: 100 images processed using streaming