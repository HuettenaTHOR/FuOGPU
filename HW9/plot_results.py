import matplotlib.pyplot as plt

# Data
num_elements = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
cpu_time = [0.231, 0.476, 0.719, 0.968, 1.215, 1.466, 1.851, 1.927, 2.187, 2.439]
gpu_time = [0.051, 0.058, 0.063, 0.072, 0.076, 0.076, 0.093, 0.087, 0.089, 0.092]

# Plot
fig, ax1 = plt.subplots()

ax1.set_xlabel("Number of Elements")
ax1.set_ylabel("CPU Time (s)")
ax1.plot(num_elements, cpu_time, marker='o', c="blue", label="CPU Time")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel("GPU Time (s)")
ax2.plot(num_elements, gpu_time, marker='s', c="red", label="GPU Time")
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("CPU vs GPU Runtime Scaling")
plt.savefig("/mnt/e/dev/FuOGPU/HW9/execution_times.png")

# Data
num_elements = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
speedup = [4.53, 8.21, 11.41, 13.44, 15.99, 19.29, 19.91, 22.15, 24.57, 26.51]

# Plot
plt.figure()
plt.plot(num_elements, speedup, marker='o')
plt.xlabel("Number of Elements")
plt.ylabel("Speedup Factor (CPU / GPU)")
plt.title("GPU Speedup over CPU")
plt.savefig("/mnt/e/dev/FuOGPU/HW9/speedup.png")
