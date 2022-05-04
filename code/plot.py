import matplotlib.pyplot as plt
import numpy as np
import argparse

# python --gt ../data/drugcell_test.txt --pred ../drugcell.predict --savefig ../res.png

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plot dcell')
	parser.add_argument('--gt', help='Input path of groudtruth data', type=str)
	parser.add_argument('--pred', help='Input path of prediction data', type=str)
	parser.add_argument('--savefig', help='Path to save the figure', type=str)
	# call functions
	arg = parser.parse_args()

	with open(arg.gt, 'r') as f:
		data = f.read().split('\n')
		x = []
		for d in data:
			if d != '':
				x.append(d.split()[2])

	with open(arg.pred, 'r') as f:
		data = f.read().split('\n')
		y = []
		for d in data:
			if d != '':
				y.append(d.split()[0])

	x = np.array(x)
	y = np.array(y)

	plt.scatter(x, y, marker='.')
	plt.xlabel("Prediction")
	plt.ylabel("Groud Truth")
	plt.xticks([])
	plt.yticks([])
	plt.title("Using unhashed fingerprint")
	plt.savefig(arg.savefig)