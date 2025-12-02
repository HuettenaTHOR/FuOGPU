# Prelims
For this HW2 I (Frederik Hüttemann, 108021215247) encoperated with my collegue (Ben Olschar, 108021211678). We did implement the code on our own but we mutually checked each others code. He also uploaded his code + results as we didn't know if uploading one result is sufficient for both. 

# Results

To generate the results, the code was run on my GPU (RTX 4060 Dual). To avoid recoding noise, the tests were run for 10.000 times. 


## Results for different matrix sizes:  
Matrix Size: 10x10, Block Size: 16x16       => CPU Avg Time: 0.237513 ms,    GPU Avg Time: 0.048213 ms  
Matrix Size: 100x100, Block Size: 16x16     => CPU Avg Time: 23.398304 ms,   GPU Avg Time: 0.048256 ms  
Matrix Size: 500x2000, Block Size: 16x16    => CPU Avg Time: 2505.173993 ms, GPU Avg Time: 0.115916 ms  
Matrix Size: 1000x1000, Block Size: 16x16   => CPU Avg Time: 2555.434394 ms, GPU Avg Time: 0.112378 ms  
Matrix Size: 100x10000, Block Size: 16x16   => CPU Avg Time: 2617.185593 ms, GPU Avg Time: 0.108724 ms  



## Results for different block sizes:  
Matrix Size: 100x10000, Block Size: 16x16 => GPU Avg Time: 0.083040 ms  
Matrix Size: 100x10000, Block Size: 16x32 => GPU Avg Time: 0.083567 ms  
Matrix Size: 100x10000, Block Size: 32x16 => GPU Avg Time: 0.080189 ms