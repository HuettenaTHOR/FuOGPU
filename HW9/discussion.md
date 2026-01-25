# Homework 9 - Discussion
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. 
We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

# Task 1: CPU Scan
We have seen, that the CPU Scan algorithm does have a time complexity of O(n). This means, with increasing number of elements, our CPU-Avg-time should grow linearly. This can be seen in the plot as well as in the table. Therefore, the results are expected.

## Task 2: GPU efficient scanning algorithm
The goal of the GPU efficient scanning algoritm was to implement a kernel, which has the same time complexity as the CPU implementation. As the CPU runs in O(n), the GPU should be able to do the same. The *Bernt-Kung algorithm* tries to implement this behavior.  
When looking at the results, we can actually see linear time complexity. Obviously the GPU in total is way faster than the CPU. However, for a growing number of elements, the execution time of the kernel grows linearly. This can be seen in both the table and the execution times plot on page 2 of the results. Here, the red graph shows the GPU execution time relative to the input size. We can also see, that both algorithms have the same time complexity as their graph lies on each other. Note, that the GPU and CPU results use different time scales for visualization purposes.