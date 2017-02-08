import random, numpy as np, math
random.seed(1)
class nodeGene:
	def __init__(self, type_, id_, value):
		self.type = type_
		self.id = id_
		self.value = value
		self.calculated = False

	def __repr__(self):
		return self.type+" : "+ str(self.id) + " Value:" + str(self.value)

class connectGene:
	def __init__(self, source, destination, weight, enabled, innovation):
		self.source = source
		self.destination = destination
		self.weight = 2 * weight - 1
		self.enabled = enabled
		self.innovation = innovation

	def __repr__(self):
		return "Source:" + str(self.source) + " destination:"+ str(self.destination) + " Weight:"+ str(self.weight) + " enabled:"+ str(self.enabled) + " innovation:"+ str(self.innovation)

class genes:
	def __init__(self, inputs, outputs):
		self.inputs = [ nodeGene("input", i, 1) for i in range(0, inputs) ]
		self.outputs = [ nodeGene("output",j, 0) for j in range(i+1, outputs + i + 1) ]
		self.nodeGenes = self.inputs + self.outputs
		self.innovation = 0
		self.connectGenes = []
		for source in self.inputs:
			for destination in self.outputs:
				self.connectGenes.append(connectGene(source.id, destination.id, random.random(), True, self.innovation))
				self.innovation += 1
		# for x in self.connectGenes:print x

class neuralNetwork:
	def __init__(self, inputs, outputs):
		self.inputs, self.outputs = inputs, outputs
		self.gnome = genes(self.inputs, self.outputs)
		self.score = 0

	def sigmoid(self , z):
		return 1.0 / ( 1.0 + math.exp(-z) )

	def calculate_backward(self, node):
		if node.calculated or node.type == "input":return node.value
		inputs = [ [ connect.source , connect.weight ] for connect in self.gnome.connectGenes if connect.destination == node.id and connect.enabled ]
		val = 0
		for inp in inputs:val += self.calculate_backward(self.gnome.nodeGenes[inp[0]]) * inp[1]
		return self.sigmoid(val)

	def forward(self, X, Y):
		if len(X[0]) != len(self.gnome.inputs):
			print "Input error"
			return
		error = []
		for i in range(len(X)):
			for j in range(len(X[i])):
				self.gnome.inputs[j].value = X[i][j] 
			for node in self.gnome.outputs:node.value = self.calculate_backward(node)
			error.append([ node.value for node in self.gnome.outputs ])
			for node in self.gnome.nodeGenes:node.value = 0
		self.score = (4 - sum(np.array(error) - Y)) ** 2
		print error,self.score

X = [ [0,0], [0,1], [1,0], [1,1] ]
Y = np.array([ [0] , [1] , [1] , [0]])
size = 10
networks = [ neuralNetwork(2,1) for i in range(size) ]
for network in networks:network.forward(X, Y)
