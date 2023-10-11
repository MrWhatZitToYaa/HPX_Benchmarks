import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import re



def comma_to_float(valstr: str):
	return float(valstr.decode("utf-8").replace(',','.'))

def extractDataFromLines(lines: str, patternName: str):
	vectorSize = []
	time = []

	numberOfIdenticalLinesWithName = 0
	lastLine = ""
	patternForNumbers = r'([+\-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)'
	prefix="/runtime"
	fullPatternForLineSearches = patternName + " Vector Size"

	for linenumber, line in enumerate(lines):
		# Count number of equal lines
		if fullPatternForLineSearches in line:
			if numberOfIdenticalLinesWithName == 0:
				number = float(re.search(patternForNumbers, line).group())
				vectorSize.append(number)
			numberOfIdenticalLinesWithName += 1
			lastLine = line
		
		# Detect a different line after several equal ones that contain name
		if fullPatternForLineSearches not in line and numberOfIdenticalLinesWithName > 0 and line.startswith(prefix):
			numberOfIdenticalLinesWithName -= 1
			# Find the largest time
			maxNumber = 0
			numbers = re.findall(patternForNumbers, line)
			number = float(numbers[-1])

			if number > maxNumber:
				maxNumber = number
			
			# Add the largest time to the list
			if numberOfIdenticalLinesWithName == 0:
				time.append(maxNumber)

	return vectorSize, time


def plotting(name: str, file_path: str, plotName: str):
	capitalName = name.capitalize()

	# Extract the data
	with open(file_path, "r") as file:
		lines = file.readlines()
	vectorSize, time = extractDataFromLines(lines, capitalName)

	plt.figure(figsize=(16, 9))
	plt.plot(vectorSize,
			time,
			marker='o',
			linestyle='none')

	# Set the size and width of axis markers and ticks for the x-axis
	plt.xticks(fontsize=14, rotation=0)  # Adjust fontsize as needed
	plt.tick_params(axis='x', which='both', direction='in', length=8, width=1.5)  # Adjust width as needed

	# Set the size and width of axis markers and ticks for the y-axis
	plt.yticks(fontsize=14)  # Adjust fontsize as needed
	plt.tick_params(axis='y', which='both', direction='in', length=8, width=1.5) 

	diagramName = name.capitalize()
	plt.title(diagramName, fontsize=20)
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel("size of vector", fontsize=14)
	plt.ylabel("time of slowest locality [s]", fontsize=14)
	plt.grid()
	plt.savefig((plotName + '.png'), dpi=400)

filePathTransform = "../Daniel/transform/scripts/"
filePathReduction = "../Daniel/reduction/scripts/"

plotting("transform", (filePathTransform + "4_nodes.txt"), "transform_4_localities")
plotting("transform", (filePathTransform + "2_nodes.txt"), "transform_2_localities")
plotting("reduction", (filePathReduction + "4_nodes.txt"), "reduction_4_localities")
#plotting("reduction", (filePathReduction + "2_nodes.txt"), "reduction_2_localities")