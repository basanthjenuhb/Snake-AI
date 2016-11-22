import pygame , random , sys , math , time , numpy as np
from pygame.locals import *

class snake:
	max_score = 0
	max_hits = 0
	pygame.init()
	i = 0
	s = pygame.display.set_mode((600, 600))
	pygame.display.set_caption('Snake')
	appleimage = pygame.Surface((20, 20))
	appleimage.fill((0, 255, 0))
	img = pygame.Surface((20, 20))
	img.fill((255, 0, 0))
	f = pygame.font.Font(None,30)
	clock = pygame.time.Clock()
	def __init__(self,net,population):
		self.xs = [290, 290, 290, 290, 290]
		self.ys = [290, 270, 250, 230, 210]
		self.dirs = 0
		self.score = 0
		self.hits = 0
		self.distance = 0
		self.net = net
		self.a , self.b = random.randint(0, 590) , random.randint(0, 590)
		self.applepos = ( self.a - self.a % 20 , self.b - self.b % 20 )
		self.population = population
		self.play()

	def collide(self,x1, x2, y1, y2, w1, w2, h1, h2):
		if x1 + w1 > x2 and x1 < x2 + w2 and y1 + h1 > y2 and y1 < y2 + h2:return True
		else:return False

	def die(self , screen , score):
		self.net.score = score
		if score > snake.max_score:
			snake.max_score = score
		if self.hits > snake.max_hits:
			snake.max_hits = self.hits
		print self.hits

	def getDistance(self):
		return math.sqrt( (self.applepos[0] - self.xs[0]) ** 2 + (self.applepos[0] - self.xs[0]) ** 2 )

	def move(self):
		# snake.clock.tick(50)
		i = len(self.xs)-1
		i = len(self.xs)-1
		while i >= 2:
			if self.collide(self.xs[0], self.xs[i], self.ys[0], self.ys[i], 20, 20, 20, 20):
				self.die(self.s, self.score)
				return True
			i-= 1
		if self.collide(self.xs[0], self.applepos[0], self.ys[0], self.applepos[1], 20, 20, 20, 20):
			self.score += 5
			self.hits += 1
			self.xs.append(700)
			self.ys.append(700)
			self.a , self.b = random.randint(0, 590) , random.randint(0, 590)
			self.applepos = ( self.a - self.a % 20 , self.b - self.b % 20 )
			self.distance = self.getDistance()
		if self.xs[0] < 0 or self.xs[0] > 580 or self.ys[0] < 0 or self.ys[0] > 580:
			self.die(self.s, self.score)
			return True
		i = len(self.xs)-1
		while i >= 1:
			self.xs[i] = self.xs[i-1]
			self.ys[i] = self.ys[i-1]
			i -= 1
		if self.dirs==0:self.ys[0] += 20
		elif self.dirs==1:self.xs[0] += 20
		elif self.dirs==2:self.ys[0] -= 20
		elif self.dirs==3:self.xs[0] -= 20	
		self.s.fill((255, 255, 255))
		for i in range(0, len(self.xs)):
			self.s.blit(self.img, (self.xs[i], self.ys[i]))
		self.s.blit(self.appleimage, self.applepos)
		t=self.f.render("Generation: "+ str(self.population.generation) + " Member: "+ str(self.population.pop.index(self.net)+1) + " max score: " + str(snake.max_score) + " Max Hits:" + str(snake.max_hits), True, (0, 0, 0))
		self.s.blit(t, (10, 10))
		t=self.f.render("Hits: " + str(self.hits) + "  Score: " + str(self.score), True, (0, 0, 0))
		self.s.blit(t, (10, 30))
		pygame.display.update()
		return False

	def select(self,x):
		if x == 0:
			self.down()
		elif x == 1:
			self.right()
		elif x == 2:
			self.up()
		else:
			self.left()

	def forward(self):
		X = self.get_input()
		results = self.net.forward(X)
		x = int(results.argmax())
		return x

	def get_input(self):
		X = np.array([ self.dirs , self.applepos[0] - self.xs[0] , self.applepos[1] - self.ys[0] ])
		X[0] = self.dirs
		return X

	def up(self):
		if self.dirs != 0:
			self.dirs = 2
		self.move()

	def down(self):
		if self.dirs != 2:
			self.dirs = 0
		self.move()

	def left(self):
		if self.dirs != 1:
			self.dirs = 3
		self.move()

	def right(self):
		if self.dirs != 3:
			self.dirs = 1
		self.move()

	def play(self):
		for i in range(2000):
			if(self.move()):
				return
			self.select(self.forward())
			if self.getDistance() < self.distance:
				self.score += 2
			else:
				self.score -= 2
			self.distance = self.getDistance()
			# pygame.image.save(snake.s , "video/screenshot" + str(snake.i) + ".jpeg")
			# snake.i += 1