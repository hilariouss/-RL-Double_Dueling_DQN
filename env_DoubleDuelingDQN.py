import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt

class GameObj():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

class GameEnv():
    def __init__(self, partial, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        a = self.reset()
        plt.imshow(a, interpolation="nearest")

    def reset(self):
        self.objects = []

        hero = GameObj(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)

        bug = GameObj(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug)

        hole = GameObj(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)

        bug2 = GameObj(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug2)

        hole2 = GameObj(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)

        bug3 = GameObj(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug3)

        bug4 = GameObj(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(bug4)

        state = self.renderEnv()
        self.state = state
        return state

    def renderEnv(self):
        render = np.ones([self.sizeY, self.sizeX, 3])
        render[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            render[item.y+1:item.y+item.size+1,  item.x+1:item.x+item.size+1,  item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            render = render[hero.y:hero.y+3, hero.x:hero.x+3,:]

        b = scipy.misc.imresize(render[:,:,0],[84, 84, 1], interp = 'nearest')
        c = scipy.misc.imresize(render[:,:,0],[84, 84, 1], interp = 'nearest')
        d = scipy.misc.imresize(render[:, :, 0], [84, 84, 1], interp='nearest')
        render = np.stack([b,c,d],axis=2)
        return render

    def moveChar(self, direction):
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize

    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameObj(self.newPosition(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(GameObj(self.newPosition(), 1, 1, 0, -1, 'fire'))
                return other.reward, False
        if ended == False:
            return 0.0, False


    def step(self, action):
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state, (reward+penalty), done
        else:
            return state, (reward+penalty), done

