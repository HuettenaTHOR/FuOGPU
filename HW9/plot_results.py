import matplotlib.pyplot as plt

# Data
num_elements = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]

cpu_time = [
    0.22859573,
    0.45950413,
    0.70350170,
    0.95229149,
    1.20651722,
    1.50089264,
    1.74140930,
    1.96731091,
    2.31409073,
    2.68909931
]

gpu_time = [
    0.24160695,
    0.17926383,
    0.18276095,
    0.17932415,
    0.18331409,
    0.62073302,
    0.64562798,
    0.63365221,
    0.62658620,
    0.64174080
]


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

speedup = [
    0.95,
    2.56,
    3.85,
    5.31,
    6.58,
    2.42,
    2.70,
    3.10,
    3.69,
    4.19
]


# Plot
plt.figure()
plt.plot(num_elements, speedup, marker='o')
plt.xlabel("Number of Elements")
plt.ylabel("Speedup Factor (CPU / GPU)")
plt.title("GPU Speedup over CPU")
plt.savefig("/mnt/e/dev/FuOGPU/HW9/speedup.png")
