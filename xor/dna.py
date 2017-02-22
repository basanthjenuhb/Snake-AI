import random, numpy as np, math, copy
# random.seed(2)
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
		self.weight = weight
		self.enabled = enabled
		self.innovation = innovation

	def __repr__(self):
		return "Source:" + str(self.source) + " destination:"+ str(self.destination) + " Weight:"+ str(self.weight) + " enabled:"+ str(self.enabled) + " innovation:"+ str(self.innovation)

class genes:
	def __init__(self, inputs, outputs):
		self.inputs = [ nodeGene("input", i, 0) for i in range(0, inputs) ]
		self.outputs = [ nodeGene("output",j, 0) for j in range(inputs , outputs + inputs) ]
		self.nodeGenes = self.inputs + self.outputs
		self.innovation = 0
		self.connectGenes = []
		for source in self.inputs:
			for destination in self.outputs:
				self.connectGenes.append(connectGene(source.id, destination.id, 20 * random.random() - 10, True, self.innovation))
				self.innovation += 1
		# for x in self.connectGenes:print x

class neuralNetwork:
	def __init__(self, inputs, outputs):
		self.inputs, self.outputs, self.total = inputs, outputs, inputs + outputs
		self.gnome = genes(self.inputs, self.outputs)
		self.score, self.output = 0, []

	@staticmethod
	def sigmoid(z):
		return 1.0 / ( 1.0 + math.exp(-z) )

	def calculate_backward(self, node):
		if node.calculated or node.type == "input":return node.value
		inputs = [ [ connect.source , connect.weight ] for connect in self.gnome.connectGenes if connect.destination == node.id and connect.enabled ]
		val = 0
		for inp in inputs:val += self.calculate_backward(self.gnome.nodeGenes[inp[0]]) * inp[1]
		node.calculated, node.value = True, self.sigmoid(val)
		return node.value

	def forward(self, X, Y, display=False):
		if len(X[0]) != len(self.gnome.inputs):
			print("Input error")
			return
		error, self.score, self.output = [], 0, []
		for i in range(len(X)):
			for node in self.gnome.nodeGenes:node.value, node.calculated = 0, False
			for j in range(len(X[i])):
				self.gnome.inputs[j].value = X[i][j] 
			for node in self.gnome.outputs:node.value = self.calculate_backward(node)
			self.output.append([ node.value for node in self.gnome.outputs ])
		self.score = sum(abs(Y - np.array(self.output))) * 10
		if display:print "The best",np.round(self.output,decimals=2).T
		# print(self.output,self.score)

	def findConnectGene(self, i):
		connectGene = [ gene for gene in self.gnome.connectGenes if gene.innovation == i ]
		if len(connectGene):return connectGene[0]

	def randomChoice(self, gene1, gene2):
		if random.random() < 0.5:
			return gene1
		return gene2

	def addDisjointGenes(self, network):
		disjointGenes = [ gene for gene in network.gnome.connectGenes if gene.innovation >= self.gnome.innovation ]
		self.gnome.connectGenes += disjointGenes

	def mate(self, network):
		child = neuralNetwork(3,1)
		if child.total < max(self.total, network.total):
			child.gnome.nodeGenes += [ nodeGene("hidden", i, 0) for i in range(child.total, child.total + ( child.total - max(self.total, network.total) )) ]
		child.gnome.connectGenes, child.gnome.innovation = [], min(self.gnome.innovation, network.gnome.innovation)
		for i in range(child.gnome.innovation):
			connectGene1, connectGene2 = self.findConnectGene(i), network.findConnectGene(i)
			child.gnome.connectGenes.append(self.randomChoice(connectGene1, connectGene2))
		dominant = self
		if self.score < network.score:dominant = network
		child.addDisjointGenes(dominant)
		return child

	def mutate(self, p1, p2):
		for connectGene in self.gnome.connectGenes:
			if random.random() < p1:
				connectGene.weight = 20 * random.random() - 10



class population:
	def __init__(self, size, X, Y):
		self.size = size
		self.X = X
		self.Y = Y
		self.networks = [ neuralNetwork(3,1) for i in range(size) ]
		self.matingPool = []
		self.bestScore, self.bestNetwork = 0, None

	def evaluateFitness(self):
		for network in self.networks:
			network.score = 0
			network.forward(self.X, self.Y)
			if network.score > self.bestScore:self.bestScore, self.bestNetwork = network.score, copy.deepcopy(network)

	def createMatingPool(self):
		self.matingPool = []
		for network in self.networks:
			for i in range(int(round(int(network.score) + 1))):
				self.matingPool.append(network)

	def newPopulation(self):
		newpop, poolSize = [], len(self.matingPool)
		for i in range(self.size):
			x, y = random.randrange(poolSize), random.randrange(poolSize)
			newpop.append(self.matingPool[x].mate(self.matingPool[y]))
		self.networks = newpop[:]

	def mutate(self, p1, p2):
		for network in self.networks:network.mutate(p1, p2)

X = [ [1,0,0], [1,0,1], [1,1,0], [1,1,1] ]
Y = np.array([[1,0,0,0]]).T
size, p1, p2 = 100, 0.2, 0.5
iterations = 20
p = population(size, X, Y)
for i in range(iterations):
	p.evaluateFitness()
	p.createMatingPool()
	p.newPopulation()
	p.mutate(p1, p2)
	print "Generation",i
print "Required",np.logical_not(Y.T).astype(int)
p.bestNetwork.forward(X,Y,True)
