import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz


def meu_filtro(fs, f1, f2, numtaps=101):
    # Insert 0 and/or 1 at the ends of cutoff so that the length of cutoff
    # is even, and each pair in cutoff corresponds to passband.
    #cutoff = np.hstack(([0.0] * pass_zero, cutoff, [1.0] * pass_nyquist))

    # `bands` is a 2-D array; each row gives the left and right edges of
    # a passband.
    #bands = cutoff.reshape(-1, 2)

    left = f1 / (fs / 2)
    right = f2 / (fs / 2)

    # Build up the coefficients.
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    print(m)
    h = 0
    h += right * np.sinc(right * m)
    h -= left * np.sinc(left * m)

    # Get and apply the window function.
    win = np.hamming(numtaps)
    h *= win

    # Now handle scaling if desired.
    # Get the first passband.
    #left, right = bands[0]
    if left == 0:
        scale_frequency = 0.0
    elif right == 1:
        scale_frequency = 1.0
    else:
        scale_frequency = 0.5 * (left + right)
    c = np.cos(np.pi * m * scale_frequency)
    s = np.sum(h * c)
    h /= s

    return h


def gera_filtro_manual(fs, f1, f2, numtaps=101):
    nyq = fs / 2
    n = np.arange(numtaps)
    fc1 = f1 / nyq
    fc2 = f2 / nyq
    
    # Ideal band-pass filter (sinc function)
    h = np.zeros(numtaps)
    center = int((numtaps - 1) / 2.0)    
    for i in range(numtaps):
        n = i - center
        if n == 0:
            h[i] = (fc2 - fc1)/np.pi
        else:
            h[i] = (np.sin(fc2 * n) - np.sin(fc1 * n)) / (np.pi * n)
    
    # Apply Hamming window
    window = np.hamming(numtaps)
    h = h * window
    
    # Normalize to ensure unity gain at center frequency
    alpha = 0.5 * (numtaps - 1)
    m = np.arange(0, numtaps) - alpha
    scale_frequency = 0.5 * (fc1 + fc2)
    c = np.cos(np.pi * m * scale_frequency)
    s = np.sum(h * c)
    h /= s
    return h

# Função para gerar o filtro passa-bandas
def gera_filtro(fs, f1, f2, numtaps=101):
    nyq = fs / 2
    taps = firwin(numtaps, [f1 / nyq, f2 / nyq], window='hamming', pass_zero=False)
    return taps

# Especificações do filtro
fs = 44100  # Frequência de amostragem
frequencias = [(10, 48), (48, 96), (96, 192), (192, 384), (384, 756), 
               (756, 1500), (1500, 3000), (3000, 6000), (6000, 12000), (12000, 20000)]

# Plotando a resposta em frequência para cada filtro
plt.figure(figsize=(14, 10))

for i, (f1, f2) in enumerate(frequencias):
    taps = meu_filtro(fs, f1, f2)
    w, h = freqz(taps, worN=8000)
    plt.subplot(5, 2, i + 1)
    plt.plot((w / np.pi) * (fs / 2), 20 * np.log10(np.abs(h)))
    plt.title(f'{f1}-{f2} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.ylim(-100, 5)
    plt.grid(True)

plt.tight_layout()
plt.show()
