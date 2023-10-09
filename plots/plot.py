import numpy as np
import matplotlib.pyplot as plt
import math

def comma_to_float(valstr: str):
    return float(valstr.decode("utf-8").replace(',','.'))

def plotting(name: str):
	data = np.loadtxt((name + '.txt'), delimiter = ",", unpack = 'true') 
	dataSize = np.array(data[0], dtype="float64")
	time = np.array(data[1], dtype="float64")
	numberOfDataPoints = dataSize.size

	plt.figure(figsize=(16, 9))

	plt.plot(dataSize,
			time,
			marker='o',
			linestyle='none')

	diagramName = name.capitalize()
	#plt.xscale("log")
	#plt.yscale("log")
	plt.xlabel("number of integers [C++ int]")
	plt.ylabel("time [s]")
	plt.title(diagramName)
	plt.grid()
	plt.savefig("(name + '.png')")


plotting("transform")
plotting("reduction")
plotting("scan")