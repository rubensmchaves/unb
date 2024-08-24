import matplotlib.pyplot as plt
import numpy as np
import math


def plot_name():
	# Create a new figure and set the window title
	plt.figure(num="My Custom Window Title")

	# Plot some data
	x = [0, 1, 2, 3, 4, 5]
	y = [0, 1, 4, 9, 16, 25]
	plt.plot(x, y)

	# Set the plot title and labels
	plt.title("Sample Plot")
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")

	# Show the plot
	plt.show()


def spectrum_test():
	# Example array
	array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

	# n value
	n = 5

	# Reading elements after each n numbers
	result = array[::n]

	print(f"Elements after each {n} numbers: {result}")
	print('remainder:', str(math.remainder(8820, 4410)))

	low = 0
	high = 0
	for i in range(len(array)):
		if i > 0 and math.remainder(i, 4) == 0: # for each 4410 samples (100 miliseconds) compute...
			low = high
			high = i
			print(array[low:high])


	SPECTRUNS = [[0,3], [3,6], [6,10], [10,20]]

	j = 0
	offset = 0
	print('array(len):', len(array))
	for j in range(len(SPECTRUNS)):
		amplitude = 0
		print('for',offset, len(array))
		for i in range(offset, len(array)):
			print('j:', j, 'i:', i, 'low:', SPECTRUNS[j][0], 'high:', SPECTRUNS[j][1])
			if i >= SPECTRUNS[j][0] and i < SPECTRUNS[j][1]:
				amplitude += array[i]
			else:
				offset = i + 1
				break
		print('amplitude:', amplitude)



if __name__ == '__main__':
	#spectrum_test()
	#plot_name()
	arr = np.array([0.00014517, 0.00014572, 0.00014664, 0.00014792])
	db = -12
	gain = 10**(db / 20)
	#gain = gain * (db/abs(db))
	print(f"ganho {gain}")	
	print(f"Antes: {arr}")
	arr = arr * gain
	print(f"Depois: {arr}")
