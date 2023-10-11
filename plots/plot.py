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


def plotting(name: str, filePath: str, fileNames: str):
	capitalName = name.capitalize()

	plt.figure(figsize=(16, 9))
	plt.xticks(fontsize=14, rotation=0)
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)
	plt.yticks(fontsize=14)
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5)
	plt.title(capitalName, fontsize=20)
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel("size of vector", fontsize=14)
	plt.ylabel("time of slowest locality [s]", fontsize=14)
	plt.grid()

	# Extract the data
	for fileName in fileNames:
		fullFilePathAndName = filePath + fileName
		with open(fullFilePathAndName, "r") as file:
			lines = file.readlines()
		vectorSize, time = extractDataFromLines(lines, capitalName)

		# Create the correct label for the data
		number = int(re.search(patternForIntegers, fileName).group())
		labelString = str(number) + " localities"

		plt.plot(vectorSize,
				 time,
				 marker='o',
				 linestyle='none',
				 label=labelString)
	
	plt.legend(fontsize=14)
	plt.savefig((name + '.png'), dpi=400)

def plot_all(filePath: str, name: str):
	searchString = "_nodes.txt"

	# Use glob to find all files that match the search string in the directory
	matchingFiles = glob.glob(os.path.join(filePath, f"*{searchString}*"))
	# Extract just the file names from the full paths
	fileNames = [os.path.basename(file) for file in matchingFiles]

	plotting(name, filePath, fileNames)



plot_all("../Daniel/transform/scripts/", "transform")
plot_all("../Daniel/reduction/scripts/", "reduction")