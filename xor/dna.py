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
	def __init__(self,weights_range, inputs, hidden, outputs):
		self.inputs = [ nodeGene("input", i, 0) for i in range(0, inputs) ]
		self.hiddens = [ nodeGene("hidden",j, 0) for j in range(inputs , hidden + inputs) ]
		self.outputs = [ nodeGene("output",j, 0) for j in range(hidden + inputs , hidden + inputs + outputs) ]
		self.nodeGenes = self.inputs + self.hiddens + self.outputs
		self.innovation = 0
		self.connectGenes = []
		for source in self.inputs:
			for destination in self.hiddens:
				self.connectGenes.append(connectGene(source.id, destination.id, random.uniform(-weights_range, weights_range), True, self.innovation))
				self.innovation += 1
		for source in self.hiddens:
			for destination in self.outputs:
				self.connectGenes.append(connectGene(source.id, destination.id, random.uniform(-weights_range, weights_range), True, self.innovation))
				self.innovation += 1
		# for x in self.connectGenes:print x

class neuralNetwork:
	def __init__(self,weights_range, inputs, hidden, outputs):
		self.inputs, self.hidden, self.outputs, self.total = inputs, hidden, outputs, inputs + hidden + outputs
		self.weights_range = weights_range
		self.gnome = genes(self.weights_range,self.inputs, self.hidden, self.outputs)
		self.score, self.output = 0, []

	@staticmethod
	def sigmoid(z):
		return 1.0 / ( 1.0 + math.exp(-10 * z) )

	def calculate_backward(self, node):
		if node.calculated or node.type == "input":
			# print node.id,node.value,node.type
			return node.value
		inputs = [ [ connect.source , connect.weight ] for connect in self.gnome.connectGenes if connect.destination == node.id and connect.enabled ]
		val = 0
		for inp in inputs:val += self.calculate_backward(self.gnome.nodeGenes[inp[0]]) * inp[1]
		node.calculated, node.value = True, self.sigmoid(val)
		# print node.id,node.value,node.type
		return node.value

	def forward(self, X, Y, display=False):
		if len(X[0]) != len(self.gnome.inputs):
			print("Input error")
			return
		error, self.score, self.output = [], 0, []
		for i in range(len(X)):
			for node in self.gnome.nodeGenes:node.value, node.calculated = 0, False
			self.gnome.hiddens[0].value, self.gnome.hiddens[0].calculated = 1, True
			for j in range(len(X[i])):
				self.gnome.inputs[j].value = X[i][j] 
			for node in self.gnome.outputs:
				node.value = self.calculate_backward(node)
			self.output.append([ node.value for node in self.gnome.outputs ])
		self.score = sum(abs(np.logical_not(Y).astype(int) - np.array(self.output))) * 10
		if display:print("The best",np.round(self.output,decimals=3).T)
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
		child = neuralNetwork(self.weights_range,*structure)
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
				if random.random() < p2:
					connectGene.weight = random.uniform(-self.weights_range,self.weights_range)
				else:
					connectGene.weight += random.uniform(-self.weights_range, self.weights_range)


class population:
	def __init__(self, structure, size, X, Y, weights_range, p1, p2):
		self.size = size
		self.X = X
		self.Y = Y
		self.weights_range, self.p1, self.p2 = weights_range, p1, p2
		self.structure = structure
		self.networks = [ neuralNetwork(weights_range,*self.structure) for i in range(size) ]
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

	def optimize(self,iterations,min_score,display):
		for i in range(iterations):
			self.evaluateFitness()
			self.createMatingPool()
			self.newPopulation()
			self.mutate(p1, p2)
			if display:print("Generation",i,"Score:",int(self.bestScore),"/",40)
			if self.bestScore > min_score:break
		# print("Required",self.Y.T)
		self.bestNetwork.forward(self.X,self.Y,display)
		print(i)
		return i

X = [ [1,0,0], [1,0,1], [1,1,0], [1,1,1] ]
Y = np.array([[0,1,1,0]]).T
size, p1, p2 = 500, 0.3, 0.5
iterations = 200
weights_range = 2
structure = (3,3,1)
min_score = 39
number_pops = 1
display = True
# p = population(structure,size, X, Y, weights_range, p1, p2,)
# p.optimize(iterations,min_score,False)
pops = [ population(structure,size, X, Y, weights_range, p1, p2,) for i in range(number_pops) ]
iterations = [ p.optimize(iterations,min_score,display) for p in pops ]
print(iterations)
