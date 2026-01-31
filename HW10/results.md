# Homework 8 - Results

## Unformatted Output
```
Running Task 0: finding optimal hyperparamters so "host-to-device", "kernel execution", and "device-to-host" each take approximately 8.00 ms
Finding optimal image size for host-to-device transfer...
Image size: 1024 x 1024, Host-to-Device time: 1.44 ms
Image size: 1024 x 1024, Device-to-Host time: 0.74 ms

Image size: 1536 x 1536, Host-to-Device time: 2.04 ms
Image size: 1536 x 1536, Device-to-Host time: 1.68 ms

Image size: 2048 x 2048, Host-to-Device time: 3.21 ms
Image size: 2048 x 2048, Device-to-Host time: 2.70 ms

Image size: 2560 x 2560, Host-to-Device time: 4.70 ms
Image size: 2560 x 2560, Device-to-Host time: 4.14 ms

Image size: 3072 x 3072, Host-to-Device time: 6.76 ms
Image size: 3072 x 3072, Device-to-Host time: 5.90 ms

Image size: 3584 x 3584, Host-to-Device time: 9.64 ms
Image size: 3584 x 3584, Device-to-Host time: 8.19 ms

Optimal image size for host-to-device transfer: 3584 x 3584 (9.64 ms)
Finding optimal kernel execution repetition time...
Repetitions: 10, Kernel execution time: 0.54 ms
Repetitions: 20, Kernel execution time: 1.10 ms
Repetitions: 30, Kernel execution time: 1.62 ms
Repetitions: 40, Kernel execution time: 2.16 ms
Repetitions: 50, Kernel execution time: 2.70 ms
Repetitions: 60, Kernel execution time: 3.24 ms
Repetitions: 70, Kernel execution time: 3.76 ms
Repetitions: 80, Kernel execution time: 4.29 ms
Repetitions: 90, Kernel execution time: 4.82 ms
Repetitions: 100, Kernel execution time: 5.03 ms
Repetitions: 110, Kernel execution time: 5.53 ms
Repetitions: 120, Kernel execution time: 5.97 ms
Repetitions: 130, Kernel execution time: 6.55 ms
Repetitions: 140, Kernel execution time: 7.08 ms
Repetitions: 150, Kernel execution time: 7.47 ms
Optimal repetitions for kernel execution: 156 (8.19 ms)
With the determined image size of 3584 x 3584, host-to-device transfer, kernel execution, and device-to-host transfer should each take approximately 8.00 ms.


Running 100 images of size 3584 x 3584 through the kernel with 156 repetitions each without streaming...
Total time without streaming for 100 images: 6316.37 ms

Running 100 images of size 3584 x 3584 through the kernel with 156 repetitions each with streaming...
Total time with streaming for 100 images: 2824.66 ms
```

## Performance Summary (Formatted Output)

| Implementation                    | Elapsed time  | Speedup        |
|-----------------------------------|---------------|----------------|  
| GPU - no streaming                | 6316.37 ms    | 1.0x           |  
| GPU - streaming                   | 2824.66 ms    | ~2.24x         |  

