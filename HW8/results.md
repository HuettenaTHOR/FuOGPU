# Homework 8 - Results

## Unformatted Output
```
Running Task 1: CPU Reduction
CPU-Avg-Time : 30.73680401ms

Running Task 2: GPU cascaded
host 10485760.00 gpu 10485760.00 
Reductions match.
GPU-Avg-Time atomic Reduction: 0.22468901ms

Running Task 3: GPU Harris
Getting Device Properties:
Number of SMs: 40
Max Threads per SM: 1024
host 10485760.00 gpu 10485760.00 
Reductions match.
GPU-Avg-Time Harris: 0.16854501ms
```

## Performance Summary (Formatted Output)

| Implementation                    | Average Time  | Speedup vs CPU |  
|-----------------------------------|---------------|----------------|  
| CPU Reduction (Sequential)        | 30.74 ms      | 1.0x           |  
| GPU Atomic Cascaded (HW7)         | 0.22 ms       | ~139.7x        |  
| GPU Harris Cascaded + threadfence | 0.17 ms       | ~180.8x        |  
