import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import re
import os
import glob

patternForScientificNumbers = r'([+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)'
patternForIntegers = r'[-]?\d+'

def comma_to_float(valstr: str):
	return float(valstr.decode("utf-8").replace(',','.'))

def extractDataFromLines(lines: str, patternName: str):
	vectorSize = []
	time = []

	numberOfIdenticalLinesWithName = 0
	lastLine = ""
	prefix="/runtime"
	fullPatternForLineSearches = patternName + " Vector Size"

	# Hotfix, only tmp
	if(fullPatternForLineSearches == "Scan Vector Size"):
		fullPatternForLineSearches = "using vector size"

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


def plottingTime(name: str, filePath: str, fileNames: str):
	capitalName = name.capitalize()

	plt.figure(figsize=(16, 9))
	plt.xticks(fontsize=14, rotation=0)
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)
	plt.yticks(fontsize=14)
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5)
	plt.title((capitalName + " computation times"), fontsize=20)
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel("size of vector", fontsize=14)
	plt.ylabel("time of slowest locality [s]", fontsize=14)
	plt.grid()

	vectorSizes = []
	times = []
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
		labelString = str(number) + " localities"
		labelStrings.append(labelString)

	for i in range(len(vectorSizes)):

		plt.plot(vectorSizes[i],
				 times[i],
				 marker='o',
				 linestyle='none',
				 label=labelStrings[i])
	
	plt.legend(fontsize=14)
	plt.savefig((name + 'Times.png'), dpi=400)

def plottingSpeedup(name: str, filePath: str, fileNames: str):
	capitalName = name.capitalize()

	plt.figure(figsize=(16, 9))
	plt.xticks(fontsize=14, rotation=0)
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)
	plt.yticks(fontsize=14)
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5)
	plt.title((capitalName + " speedup"), fontsize=20)
	plt.xscale("log")
	#plt.yscale("log")
	plt.xlabel("size of vector", fontsize=14)
	plt.ylabel("speedup", fontsize=14)
	plt.grid()

	vectorSizes = []
	times = []
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
		labelString = str(number) + " localities"
		labelStrings.append(labelString)

		if(number == 1):
			locality1VectorSize = vectorSize
			locality1Time = time


	for i in range(len(vectorSizes)):
		speedup = [b / a for a, b in zip(times[i], locality1Time)]
		
		plt.plot(vectorSizes[i],
				 speedup,
				 marker='o',
				 linestyle='none',
				 label=labelStrings[i])
	
	plt.legend(fontsize=14)
	plt.savefig((name + 'Speedup.png'), dpi=400)

def plot_all(filePath: str, name: str):
	searchString = "_nodes.txt"

	# Use glob to find all files that match the search string in the directory
	matchingFiles = glob.glob(os.path.join(filePath, f"*{searchString}*"))
	# Extract just the file names from the full paths
	fileNames = [os.path.basename(file) for file in matchingFiles]

	if(len(fileNames) == 0):
		raise ValueError("Data not found")

	plottingTime(name, filePath, fileNames)
	plottingSpeedup(name, filePath, fileNames)



#plot_all("../Daniel/transform/measurements/", "transform")
#plot_all("../Daniel/reduction/measurements/", "reduction")
plot_all("../Daniel/scan/measurements/", "scan")