# Results
For this Ben Olschar, 108021211678 and Frederik Hüttemann, 108021215247 cooperated with eachother. We did implement the code on our own, but agreed to using one version. The discussion and results are made in cooperation.

## Console output
~~~
Running Task 1: CPU matrix transpose
CPU transpose: avg. time: 327.357621 ms, bandwidth: 1.137988 GB per second
Running Task 2: GPU row to row copy
Results match.
GPU row to row copy: avg. time: 0.301871 ms, bandwidth: 1234.066718 GB per second
Running Task 3: GPU naive matrix transpose
Results match.
GPU naive transpose: avg. time: 1.363735 ms, bandwidth: 273.168204 GB per second
Running Task 4: GPU shared Memory transpose (without padding) 
Results match.
GPU shared memory transpose (without padding): avg. time: 0.700969 ms, bandwidth: 531.448666 GB per second
Running Task 4: GPU shared Memory transpose 
Results match.
GPU shared memory transpose (with padding --> no bank conflicts): avg. time: 0.213564 ms, bandwidth: 1744.344417 GB per second 
~~~ 

## Formatted output
| Routine                                                     | RTX 4060ti - Time (ms) | RTX 4060ti - Bandwidth (GB/s) |
|-------------------------------------------------------------|------------------------|-------------------------------|
| CPU transpose                                               | 327,358 ms             | 1,138 GB / s                  |
| GPU row2row copy                                            | 0,302 ms               | 1234,067 GB / s               |
| GPU naive transpose                                         | 1,364 ms               | 273,168 GB / s                |
| GPU shared mem transpose (with bank conflicts)              | 0,701 ms               | 531,449 GB / s                |
| GPU shared mem transpose (with padding / no bank conflicts) | 0,214 ms               | 1744,344 GB / s               |