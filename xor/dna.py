import random

class nodeGene:
	def __init__(self, type_, id_, value):
		self.type = type_
		self.id = id_
		self.value = value

	def __str__(self):
		return str(self.type)+" : "+ str(self.id) + "Value:" + str(self.value)

class connectGene:
	def __init__(self, source, destination, weight, enabled, innovation):
		self.source = source
		self.destination = destination
		self.weight = 2 * weight - 1
		self.enabled = enabled
		self.innovation = innovation

	def __str__(self):
		return "Source:" + str(self.source) + ", destination:"+ str(self.destination) + ", Weight:"+ str(self.weight) + ", enabled:"+ str(self.enabled) + ", innovation:"+ str(self.innovation)

class genes:
	def __init__(self, inputs, outputs):
		self.inputs = [ nodeGene("input", i, 0) for i in range(0, inputs) ]
		self.hidden = []
		self.outputs = [ nodeGene("output",j, 0) for j in range(i+1, outputs + i + 1) ]
		self.nodeGenes = self.inputs + self.outputs
		self.innovation = 0
		self.connectGenes = []
		for source in self.inputs:
			for destination in self.outputs:
				self.connectGenes.append(connectGene(source.id, destination.id, random.random(), True, self.innovation))
				self.innovation += 1
		for x in self.connectGenes:print str(x)

class neuralNetwork:
	def __init__(self, inputs, outputs):
		self.inputs, self.outputs = inputs, outputs
		self.gnome = genes(self.inputs, self.outputs)

	def forward(self, x):
		for i in range(len(x)):self.gnome.nodeGenes[i].value = x[i]
		# for connect in self.gnome.connectGenes:


neuralNetwork(4,2)
