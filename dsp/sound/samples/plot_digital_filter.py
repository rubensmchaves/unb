import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Example filter coefficients
b = [0.1, 0.15, 0.5, 0.15, 0.1]  # FIR filter coefficients (numerator)
a = 1  # FIR filter, so a is just 1

# Compute the frequency response
w, h = freqz(b, a, worN=8000)

# Plot the magnitude response
plt.figure()
plt.plot(w / np.pi, 20 * np.log10(np.abs(h)))
plt.title('Magnitude Response of the Filter')
plt.xlabel('Normalized Frequency [π*rad/sample]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()
