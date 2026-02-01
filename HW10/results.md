# Homework 10 - Results

## Task 0: Execution Time Measurements (Full HD: 1920 x 1080)

| Operation | Execution Time |
|-----------|----------------|
| Host-to-Device transfer | 1.44 ms |
| Kernel execution | 1.47 ms |
| Device-to-Host transfer | 1.35 ms |

The three phases are approximately balanced (~1.4-1.5 ms each), enabling effective overlap with CUDA streams.

## Task 1 & 2: Processing 100 Full HD Images

### Unformatted Output
```
Task 0: Measuring execution times for Full HD images (1920 x 1080)

Host-to-Device transfer time: 1.44 ms (avg of 10 runs)
Kernel execution time: 1.47 ms (avg of 10 runs)
Device-to-Host transfer time: 1.35 ms (avg of 10 runs)

Summary: H2D=1.44ms, Kernel=1.47ms, D2H=1.35ms

Task 1: Processing 100 Full HD images without streaming (10 runs for averaging)

  Run 1: 418.32 ms
  Run 2: 415.23 ms
  Run 3: 414.29 ms
  Run 4: 414.25 ms
  Run 5: 418.46 ms
  Run 6: 413.88 ms
  Run 7: 414.02 ms
  Run 8: 414.19 ms
  Run 9: 416.08 ms
  Run 10: 421.61 ms

Average time (no streaming): 416.03 ms

Task 2: Processing 100 Full HD images WITH streaming (4 streams, 10 runs for averaging)

  Run 1: 570.15 ms (warmup)
  Run 2: 379.15 ms
  Run 3: 378.72 ms
  Run 4: 380.82 ms
  Run 5: 379.63 ms
  Run 6: 380.44 ms
  Run 7: 379.88 ms
  Run 8: 379.48 ms
  Run 9: 381.05 ms
  Run 10: 380.99 ms

Average time WITH streaming: 380.02 ms (excluding warmup)
```

## Performance Summary

| Implementation | Average Time | Speedup |
|----------------|--------------|---------|
| GPU - no streaming | 416.03 ms | baseline |
| GPU - with streaming (4 streams) | 380.02 ms | **1.095x (9.5%)** |  

