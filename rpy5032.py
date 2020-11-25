########################################################
#
# CMPSC 441: Homework 4
#
########################################################



student_name = 'Type your full name here'
student_email = 'Type your email address here'



########################################################
# Import
########################################################

from hw4_utils import *
import math
import random



# Add your imports here if used






################################################################
# 1. Genetic Algorithm
################################################################


def genetic_algorithm(problem, f_thres, ngen=1000):
    """
    Returns a tuple (i, sol) 
    where
      - i  : number of generations computed
      - sol: best chromosome found
    """
    population = problem.init_population()
    best = problem.fittest(population, f_thres)
    if best != None:
        return (-1, best)
    for i in range(ngen):
        population = problem.next_generation(population)
        best = problem.fittest(population, f_thres)
        if best != None:
            return (i, best)
    best = problem.fittest(population)
    return (ngen, best)

################################################################
# 2. NQueens Problem
################################################################


class NQueensProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob
    
    def init_population(self):
        pop = []
        for i in range(self.n):
            chrom = []
            for j in range(self.g_len):
                chrom.append(random.choice(self.g_bases))
            pop.append(tuple(chrom))
        return pop
 
    def next_generation(self, population):
        pop = []
        for i in range(self.n):
            parent = self.select(2, population)
            child = self.crossover(parent[0], parent[1])
            pop.append(self.mutate(child))
        return pop

    def mutate(self, chrom):
        p = random.random()
        if(p > self.m_prob):
            return chrom
        else:
            i = random.randrange(self.g_len)
            chrom2 = list(chrom)
            chrom2[i] = random.choice(self.g_bases)
            return tuple(chrom2)
    
    def crossover(self, chrom1, chrom2):
        i = random.randrange(self.g_len)
        chrom = chrom1[:i] + chrom2[i:]
        return chrom

    def fitness_fn(self, chrom):
        cnt = 0;
        for c1, r1 in enumerate(chrom):
            for c2, r2 in enumerate(chrom):
                if c2 > c1:
                    if (r1, c1) != (r2, c2) and not self.conflict(r1, c1, r2, c2):
                        cnt = cnt + 1
        return cnt

    def conflict(self, row1, col1, row2, col2):
        return row1 == row2 or col1 == col2 or abs(row1-row2) == abs(col1-col2)
        
    def select(self, m, population):
        pop = []
        fit_sum = 0
        for j in population:
            fit_sum = fit_sum + self.fitness_fn(j)
        fit = []
        for k in population:
            fit.append(self.fitness_fn(k)/fit_sum)
        dist = []
        prob = 0
        for l in fit:
            prob = prob + l
            dist.append(prob)
        for i in range(m):
            p = random.random()
            chrom = None
            for m in range(len(dist)):
                if p > dist[m]:
                    continue
                else:
                    chrom = population[m]
                    break
            pop.append(chrom)
        return pop
    
    def fittest(self, population, f_thres=None):
        best = population[0]
        if f_thres != None:
            if all(self.fitness_fn(element) < f_thres for element in population):
                return None
        for i in population:
                if self.fitness_fn(i) > self.fitness_fn(best):
                    best = i
        return best
            




        
################################################################
# 3. Function Optimaization f(x,y) = x sin(4x) + 1.1 y sin(2y)
################################################################


class FunctionProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob

    def init_population(self):
        pop = []
        for i in range(self.n):
            pop.append((random.uniform(0, self.g_bases[0]),random.uniform(0, self.g_bases[1])))
        return pop

    def next_generation(self, population):
        pop = []
        m = population[:]
        for i in range(int(self.n/2)):
            pop.append(self.fittest(m))
            m.remove(self.fittest(m))
        for j in range(int(self.n/2) + 1):
            parent = self.select(2, pop)
            child = self.crossover(parent[0], parent[1])
            pop.append(self.mutate(child))
        #print(len(pop))
        #print(self.n)
        return pop
        
    def mutate(self, chrom):
        p = random.random()
        if(p > self.m_prob):
            return chrom
        else:
            i = random.randrange(len(chrom))
            chrom2 = list(chrom)
            chrom2[i] = random.uniform(0, self.g_bases[i])
            return tuple(chrom2)
        
    def crossover(self, chrom1, chrom2):
        a = random.random()
        n = random.randrange(0,2)
        if(n == 0):
            xnew = (1 - a) * chrom1[0] + a * chrom2[0]
            return (xnew, chrom1[1])
        else:
            ynew = (1 - a) * chrom1[1] + a * chrom2[1]
            return (chrom1[0], ynew)
    
    def fitness_fn(self, chrom):
        return chrom[0] * math.sin(4 * chrom[0]) + 1.1 * chrom[1] * math.sin(2 * chrom[1])

    
    def select(self, m, population):
        adict = {}
        ranked = {}
        for i in population:
            adict[self.fitness_fn(i)] = i
        n = 0
        for j in sorted(adict):
            ranked[n] = adict[j]
            n = n+1
        fit = []
        for k in ranked.keys():
            #print(k)
            #print(len(population))
            fit.append((len(population) - k)/((len(population) + 1) * len(population)/2))
        dist = []
        prob = 0
        for l in fit:
            prob = prob + l
            dist.append(prob)
        #print(dist)
        pop = []
        for i in range(m):
            p = random.random()
            chrom = population[0]
            for m in range(len(dist)):
                if p > dist[m]:
                    continue
                else:
                    chrom = population[m]
                    break
            #print(p)
            pop.append(chrom)
        return pop

    def fittest(self, population, f_thres=None):
        best = population[0]
        if f_thres != None:
            if all(self.fitness_fn(element) > f_thres for element in population):
                return None
        for i in population:
                if self.fitness_fn(i) < self.fitness_fn(best):
                    best = i
        return best




################################################################
# 4. Traveling Salesman Problem
################################################################


class HamiltonProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob, graph=None):
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob
        self.graph = graph

    def init_population(self):
        pop = []
        for i in range(self.n):
            x = random.sample(self.g_bases, len(self.g_bases))
            pop.append(x)
        return pop
          
    def next_generation(self, population):
        pop = []
        children = []
        for i in range(self.n):
            parent = self.select(2, population)
            child = self.crossover(parent[0], parent[1])
            children.append(self.mutate(child))
        population = population + children
        for j in range(self.n):
            pop.append(self.fittest(population))
            population.remove(self.fittest(population))
        return pop
          
    def mutate(self, chrom):
        p = random.random()
        if(p > self.m_prob):
            return chrom
        else:
            i = random.randrange(len(chrom))
            j = random.randrange(len(chrom))
            chrom2 = list(chrom)
            temp = chrom2[i]
            chrom2[i] = chrom2[j]
            chrom2[j] = temp
            return tuple(chrom2)
    
    def crossover(self, chrom1, chrom2):
        i = random.randrange(len(chrom1))
        #print(i)
        chr1 = list(chrom1)
        chr2 = list(chrom2)
        #print(chr1[i], chr2[i])
        chr1[i] = chr2[i]
        #print(chr1)
        while(len(set(chr1)) != len(set(chr2))):
            for j in range(len(chr1)):
                if chr1[j] == chr1[i] and i != j:
                    chr1[j] = chr2[j]
                    #print(chr1)
                    i = j
        return tuple(chr1)

    def fitness_fn(self, chrom):
        dist = 0
        for i in range(len(chrom) - 1):
            dist = dist + self.graph.get(chrom[i], chrom[i+1])
        dist = dist + self.graph.get(chrom[len(chrom) - 1], chrom[0])
        return dist
          

    def select(self, m, population):
        T = 0
        for i in population:
            T = T + self.fitness_fn(i)
        fit_sum = 0
        for j in range(len(population)):
            fit_sum = fit_sum + T - self.fitness_fn(population[j])
        fit = []
        for k in population:
            fit.append((T - self.fitness_fn(k))/fit_sum)
        dist = []
        prob = 0
        for l in fit:
            prob = prob + l
            dist.append(prob)
        pop = []
        for i in range(m):
            p = random.random()
            chrom = None
            for m in range(len(dist)):
                if p > dist[m]:
                    continue
                else:
                    chrom = population[m]
                    break
            pop.append(chrom)
        return pop
        
        
    def fittest(self, population, f_thres=None):
        best = population[0]
        if f_thres != None:
            if all(self.fitness_fn(element) > f_thres for element in population):
                return None
        for i in population:
                if self.fitness_fn(i) < self.fitness_fn(best):
                    best = i
        return best
