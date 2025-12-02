# Prelims
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. We did implement the code on our own, but agreed to using one version. The discussion is formulated together.

# Results
This file contains the formatted output from the code.

## Task 1: "only implementation"
nothing to see

## Task 2: "compare shared vs non-shared algorithms on both CPU and GPU"
### CPU: (repeated 10 times for average)
Average CPU Shared Memory Time: 14562.203649 s  
Average CPU Non-Shared Memory Time: 11574.565241 s  
Results match.

### GPU (repeated 100 times for average)
Average GPU Shared Memory Time: 1.232409 s  
Average GPU Non-Shared Memory Time: 1.944097 s  
Results match.

## Task 3: "compare execution time for host and device (CPU and GPU) implementations"
Average CPU Shared Memory Time: 14562.203649 s  
Average GPU Shared Memory Time: 1.232409 s  


## Task 4.1: "Show dependency of the execution time on TILEWIDTH. Determine the parameters for the shortest execution time"
GPU Time with TILE_WIDTH=4: 8.941602 s  
GPU Time with TILE_WIDTH=6: 4.591954 s  
GPU Time with TILE_WIDTH=8: 4.717529 s  
GPU Time with TILE_WIDTH=10: 3.147476 s  
GPU Time with TILE_WIDTH=12: 2.606590 s  
GPU Time with TILE_WIDTH=14: 2.306206 s  
GPU Time with TILE_WIDTH=16: 1.707149 s  
GPU Time with TILE_WIDTH=18: 1.684711 s  
GPU Time with TILE_WIDTH=20: 1.562358 s  
GPU Time with TILE_WIDTH=22: 1.419778 s  
GPU Time with TILE_WIDTH=24: 1.340641 s  
GPU Time with TILE_WIDTH=26: 1.443911 s  
GPU Time with TILE_WIDTH=28: 1.627100 s  
GPU Time with TILE_WIDTH=30: 1.675427 s  
GPU Time with TILE_WIDTH=32: 1.307674 s  

There is also a plot given which show the dependence on TILEWIDTH.

## Task 4.2: "Study the dependence of the of speedup factor on matrix size"
Note: for this the resulting dimensions were multiplied to quantify the matrix size  

Matrix size: M(100 x 50), N(50 x 100)  
CPU Time: 0.002232 s  
GPU Time: 0.000047 s  
Speedup (10000 elements in result): (CPU time / GPU time): 47.299550  

Matrix size: M(500 x 250), N(250 x 500)  
CPU Time: 0.290189 s  
GPU Time: 0.000127 s  
Speedup (250000 elements in result): (CPU time / GPU time): 2290.183645  

Matrix size: M(1000 x 500), N(500 x 1000)  
CPU Time: 2.491026 s  
GPU Time: 0.002296 s  
Speedup (1000000 elements in result): (CPU time / GPU time): 1085.173960  

Matrix size: M(2000 x 1000), N(1000 x 2000)  
CPU Time: 26.466413 s  
GPU Time: 0.006593 s  
Speedup (4000000 elements in result): (CPU time / GPU time): 4014.112072  

Matrix size: M(5000 x 2500), N(2500 x 5000)  
CPU Time: 780.202429 s  
GPU Time: 0.081067 s  
Speedup (25000000 elements in result): (CPU time / GPU time): 9624.140917  

There is also a plot given which show the relation on speedup vs matrix size

