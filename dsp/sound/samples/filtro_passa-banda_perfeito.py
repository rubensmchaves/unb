import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

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
    taps = gera_filtro(fs, f1, f2)
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
