import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import re
import os
import glob
from numpy import argsort

patternForScientificNumbers = r'([+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)'
patternForIntegers = r'[-]?\d+'

def comma_to_float(valstr: str):
	return float(valstr.decode("utf-8").replace(',','.'))

def extractDataFromLinesWeird(lines: str, patternName: str):
	vectorSize = []
	time = []

	numberOfIdenticalLinesWithName = 0
	lastLine = ""
	prefix="/runtime"
	fullPatternForLineSearches = patternName + " Vector Size"

	for linenumber, line in enumerate(lines):
		# Count number of equal lines
		if fullPatternForLineSearches in line:
			if numberOfIdenticalLinesWithName == 0:
				number = float(re.search(patternForScientificNumbers, line).group())
				vectorSize.append(number)
			numberOfIdenticalLinesWithName += 1
			lastLine = line
		
		# Detect a different line after several equal ones that contain name
		if fullPatternForLineSearches not in line and numberOfIdenticalLinesWithName > 0 and line.startswith(prefix):
			numberOfIdenticalLinesWithName -= 1
			# Find the largest time
			maxNumber = 0
			numbers = re.findall(patternForScientificNumbers, line)
			number = float(numbers[-1])

			if number > maxNumber:
				maxNumber = number
			
			# Add the largest time to the list
			if numberOfIdenticalLinesWithName == 0:
				time.append(maxNumber)

	return vectorSize, time

def extractDataFromLinesCSV(lines: str, patternName: str):
	vectorSize = []
	time = []

	for i, line in enumerate(lines):
		if(i==0):
			pass
		else:
			# Split the line into values using a comma as the delimiter
			values = line.strip().split(',')
			vectorSize.append(float(values[0]))

			tmpTime = [float(item) for item in values[1:]]
			mean = sum(tmpTime) / len(tmpTime)
			time.append(mean)
	return vectorSize, time

def extractDataFromLines(lines: str, patternName: str):
	#return extractDataFromLinesWeird(lines, patternName)
	return extractDataFromLinesCSV(lines, patternName)

def plottingTime(name: str, filePath: str, fileNames: str, clusterName: str):
	capitalName = name.capitalize()
	clusterName = clusterName.capitalize()

	plt.figure(figsize=(16, 9))
	plt.xticks(fontsize=14, rotation=0)
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)
	plt.yticks(fontsize=14)
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5)
	plt.title((capitalName + " computation times on " + clusterName), fontsize=20)
	plt.xscale("log", base=2)
	plt.yscale("log", base=10)
	plt.xlabel("Number of elements in vector", fontsize=14)
	plt.ylabel("Time of slowest locality [s]", fontsize=14)
	plt.grid()

	vectorSizes = []
	times = []
	numbers = []
	labelStrings = []
	# Extract the data and save it into a 2D list
	for fileName in fileNames:
		fullFilePathAndName = filePath + fileName
		with open(fullFilePathAndName, "r") as file:
			lines = file.readlines()
		vectorSize, time = extractDataFromLines(lines, capitalName)
		vectorSizes.append(vectorSize)
		times.append(time)

		# Create the correct label for the data
		number = int(re.search(patternForIntegers, fileName).group())
		numbers.append(number)
		labelString = str(number) + " localities"
		labelStrings.append(labelString)

	# Plot the results in asending order of #localyties
	numberIndices = argsort(numbers)
				  
	for i in numberIndices:
		plt.plot(vectorSizes[i],
				 times[i],
				 marker='o',
				 linestyle='none',
				 label=labelStrings[i])
	
	plt.legend(fontsize=14)
	plt.savefig((filePath + name + clusterName + 'Times.png'), dpi=400)

def plottingSpeedup(name: str, filePath: str, fileNames: str, clusterName: str):
	capitalName = name.capitalize()
	clusterName = clusterName.capitalize()

	plt.figure(figsize=(16, 9))
	plt.xticks(fontsize=14, rotation=0)
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)
	plt.yticks(fontsize=14)
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5)
	plt.title((capitalName + " speedup on " + clusterName), fontsize=20)
	plt.xscale("log", base=2)
	#plt.yscale("log")
	plt.xlabel("Number of elements in vector", fontsize=14)
	plt.ylabel("Speedup T[1] / T[n]", fontsize=14)
	plt.grid()

	vectorSizes = []
	times = []
	numbers = []
	labelStrings = []

	locality1VectorSize = []
	locality1Time = []
	# Extract the data and save it into a 2D list
	for fileName in fileNames:
		fullFilePathAndName = filePath + fileName
		with open(fullFilePathAndName, "r") as file:
			lines = file.readlines()
		vectorSize, time = extractDataFromLines(lines, capitalName)
		vectorSizes.append(vectorSize)
		times.append(time)

		# Create the correct label for the data
		number = int(re.search(patternForIntegers, fileName).group())
		numbers.append(number)
		labelString = str(number) + " localities"
		labelStrings.append(labelString)

		if(number == 1):
			locality1VectorSize = vectorSize
			locality1Time = time

	# Plot the results in asending order of #localyties
	numberIndices = argsort(numbers)
				  
	for i in numberIndices:
		speedup = [b / a for a, b in zip(times[i], locality1Time)]
		
		plt.plot(vectorSizes[i],
				 speedup,
				 marker='o',
				 linestyle='none',
				 label=labelStrings[i])
	
	plt.ylim(bottom=0)
	plt.legend(fontsize=14)
	plt.savefig((filePath + name + clusterName + 'Speedup.png'), dpi=400)

def plottingBandwith(name: str, filePath: str, fileNames: str, bytesPerNumber: int, clusterName: str):
	capitalName = name.capitalize()
	clusterName = clusterName.capitalize()

	plt.figure(figsize=(16, 9))
	plt.xticks(fontsize=14, rotation=0)
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)
	plt.yticks(fontsize=14)
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5)
	plt.title((capitalName + " bandwidth on " + clusterName), fontsize=20)
	plt.xscale("log", base=2)
	#plt.yscale("log", base=2)
	plt.xlabel("Number of elements in vector", fontsize=14)
	plt.ylabel("Bandwidth [GB/s]", fontsize=14)
	plt.grid()

	vectorSizes = []
	times = []
	numbers = []
	labelStrings = []
	# Extract the data and save it into a 2D list
	for fileName in fileNames:
		fullFilePathAndName = filePath + fileName
		with open(fullFilePathAndName, "r") as file:
			lines = file.readlines()
		vectorSize, time = extractDataFromLines(lines, capitalName)
		vectorSizes.append(vectorSize)
		times.append(time)

		# Create the correct label for the data
		number = int(re.search(patternForIntegers, fileName).group())
		numbers.append(number)
		labelString = str(number) + " localities"
		labelStrings.append(labelString)

	# Plot the results in asending order of #localyties
	numberIndices = argsort(numbers)
	
	# Transform uses 2 vectors, so double the data
	multiplicator = 1
	if name == "transform":
		multiplicator = 2

	for i in numberIndices:
		bandwidth = [a * bytesPerNumber * multiplicator / b for a, b in zip(vectorSizes[i], times[i])]
		bandwidthInGB = [i * 1e-9 for i in bandwidth]
		plt.plot(vectorSizes[i],
				 bandwidthInGB,
				 marker='o',
				 linestyle='none',
				 label=labelStrings[i])
	
	plt.legend(fontsize=14)
	plt.savefig((filePath + name + clusterName + 'Bandwidth.png'), dpi=400)

def plot_all(filePath: str, name: str, clusterName: str):
	startSequence  = clusterName
	endSequence   = ".txt"

	matchingFiles = []
	for fullFilename in glob.glob(os.path.join(filePath, "*")):
		fileName = os.path.basename(fullFilename)
		
		if re.match(f"{startSequence}.*{endSequence}$", fileName):
			matchingFiles.append(fileName)
	
	# Extract just the file names from the full paths
	fileNames = [os.path.basename(file) for file in matchingFiles]

	if(len(fileNames) == 0):
		raise ValueError("Data not found")

	plottingTime(name, filePath, fileNames, clusterName)
	print("Done plotting time for", name, clusterName)
	plottingSpeedup(name, filePath, fileNames, clusterName)
	print("Done speedup time for", name, clusterName)
	plottingBandwith(name, filePath, fileNames, 4, clusterName)
	print("Done bandwidth time for", name, clusterName)



plot_all("./finalPlots/transform/", "transform", "qdr")
#plot_all("./finalPlots/transform/", "transform", "rome")
plot_all("./finalPlots/reduction/", "reduction", "qdr")
#plot_all("./finalPlots/reduction/", "reduction", "rome")
plot_all("./finalPlots/scan/", "scan", "qdr")
#plot_all("./finalPlots/scan/", "scan", "rome")