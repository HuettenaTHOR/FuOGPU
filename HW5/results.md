# Results
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

## Console output
~~~
Running Task 1: CPU matrix transpose
CPU transpose: avg. time: 310.651729 ms, bandwidth: 0.643808 GB per second
Running Task 2: GPU row to row copy
Results match.
GPU row to row copy: avg. time: 0.254130 ms, bandwidth: 786.998393 GB per second
Running Task 3: GPU naive matrix transpose
Results match.
GPU naive transpose: avg. time: 1.232887 ms, bandwidth: 162.220865 GB per second
Running Task 4: GPU shared Memory transpose (without padding) 
Results match.
GPU shared memory transpose (without padding): avg. time: 0.646181 ms, bandwidth: 309.510752 GB per second
Running Task 4: GPU shared Memory transpose 
Results match.
GPU shared memory transpose (with padding --> no bank conflicts): avg. time: 0.208878 ms, bandwidth: 957.496536 GB per second
~~~ 

## Formatted output
| Routine                                                     | RTX 4060ti - Time (ms) | RTX 4060ti - Bandwidth (GB/s) |
|-------------------------------------------------------------|------------------------|-------------------------------|
| CPU transpose                                               | 310,65 ms              | 0,6438 GB / s                 |
| GPU row2row copy                                            | 0,254 ms               | 786,998 GB / s                |
| GPU naive transpose                                         | 1,233 ms               | 162,223 GB / s                |
| GPU shared mem transpose (with bank conflicts)              | 0,646 ms               | 309,511 GB / s                |
| GPU shared mem transpose (with padding / no bank conflicts) | 0,209 ms               | 957,497 GB / s                |