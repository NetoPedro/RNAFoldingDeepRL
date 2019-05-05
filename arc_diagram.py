"""
Very simple application  for vizualizing the arc diagrams
@ Mehdi Saman Booy
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc, Circle
import numpy as np
from sys import argv


def _arc(i, j, width=1, linestyle='-', color='black'):
	"""
	Creating a single arc from i to j
	"""
	return Arc(((i+j)/2., 0), abs(i-j), abs(i-j), 0, 0, 180, linewidth=width, 
		edgecolor=color, fill=False, linestyle=linestyle)

def _circle(i, r=.05):
	"""
	Create a small filled circle with center at (i, 0) and radius r
	"""
	return Circle((i, 0), r, fill=True, color='black')

def arc_diagram(x, linestyle='-', color='black', width=.5, self_loop='same'):
	"""
	self_loop (str): 'same' means you are showing self-loop of i with i
					 '-1' means you are showing self0loop of i with -1
	"""
	plt.clf()
	ax = plt.gca()
	
	plt.plot([0, len(x)-1], [0, 0], color='black', linewidth=.7)
	plt.axis('off')
	for i in range(len(x)):
		j = x[i]
		ax.add_patch(_circle(i))
		sl_val = -1 if self_loop=='-1' else i
		if j != sl_val:
			c = _arc(i, j, width=width, linestyle=linestyle, color=color)
			ax.add_patch(c)
	
	plt.axis('scaled')
	return ax
	

def phrantheses_to_pairing_list(str, self_loop='same'):
	N = len(str)
	pairing = [0] * N
	stack = []
	for (i, s) in enumerate(str):
		if s == ')':
			j = stack[-1]
			stack = stack[:-1]
			pairing[i] = j
			pairing[j] = i
		elif s == '(':
			stack.append(i)
		else:
			sl_val = -1 if self_loop=='-1' else i
			pairing[i] = sl_val
	return list(pairing)


if __name__ == '__main__':
	d1 = '((((....))))'
	d2 = '..((((..))))'

	for i in [d1, d2]:
		plt.figure()
		ax = arc_diagram(phrantheses_to_pairing_list(i), width=.8, linestyle='--')
	
	plt.show()
	