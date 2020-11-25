########################################################
#
# CMPSC 441: Homework 4
#
########################################################


#
# DO NOT CHANGE ANYTHING IN THIS FILE.
# - Your submission will be tested using this file
#   in its original contents.
#


import math


class GeneticProblem:
    def __init__(self, n, g_bases, g_len, m_prob):
        """
        Initializes attributes of the class
        Arguments:
        - n       : number of chromosomes in the population
        - g_bases : gene bases
        - g_len   : length of a chromosome
        - m_prob  : mutation probability
        """
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob
    

    def init_population(self):
        """
        Returns an initial population. Initial population is a list
        that contains n randomly generated chromosomes. Each chromosome
        is a tuple of g_len items selected from g_bases
        """
        pass

    
    def next_generation(self, population):
        """
        Returns the next generation of population containing n chromosomes
        obtained by applying crossover and mutation to the given population.
        """
        pass


    def mutate(self, chrom):
        """
        Returns a chromosome after mutating the given chrom at a 
        random position with the probability of m_prob
        """
        pass

    
    def crossover(self, chrom1, chrom2):
        """
        Returns an offspring obtained by applying crossover to the 
        two given chromosomes, chrom1 and chrom2. If the crossover
        occurs at a random index i, then the offspring is created 
        by combining chrom1[:i] with chrom2[i:]
        """
        pass


    def fitness_fn(self, chrom):
        """Returns the fitness value of the given chrom"""
        pass


    def select(self, m, population):
        """
        Returns a list of m chromosomes randomly selected from the
        given population using fitness proportionate selection.
        """
        pass


    def fittest(self, population, f_thres=None):
        """
        If f_thres is None, return the best chromosome in the given population.
        If f_thres is not None, return the best chromosome if its fitness value
        is better than f_thres. Otherwise, return None.
        """
        pass



class Graph:
    def __init__(self, edges=None, directed=True):
        self.edges = edges or {}
        self.directed = directed
        if not directed:
            for x in list(self.edges.keys()):
                for (y, dist) in self.edges[x].items():
                    self.edges.setdefault(y,{})[x] = dist

    def get(self, x, y=None):
        """Returns the distance to y from x, or
            the distance to cities reachable from x"""
        edges = self.edges.setdefault(x,{})
        if y is None:
            return edges
        else:
            return edges.get(y, math.inf)

    def vertices(self):
        """Returns the list of vertices in the graph."""
        s = set([x for x in self.edges.keys()])
        t = set([y for v in self.edges.values() for (y,d) in v.items()])
        v = s.union(t)
        return list(v)

    def __repr__(self):
        return "<Graph {}>".format(self.edges)




univ_bases = ['Arizona_State', 'Brigham_Young', 'Brown', 'Colorado',
              'Duke', 'Florida_State', 'Louisiana', 'Louisville',
              'Michigan', 'New_Mexico', 'North_Dakota', 'Notre_Dame',
              'Ohio', 'Oklahoma', 'Oregon', 'Pittsburgh', 'Stanford',
              'Texas_AM', 'Wisconsin', 'Yale']

univ_roads = dict(
    Arizona_State=dict(Brigham_Young= 648, Brown=2625, Colorado= 549,
                       Duke=2185, Florida_State =1898, Louisiana =1458,
                       Louisville=1752, Michigan=1963, New_Mexico= 427,
                       North_Dakota=1743, Notre_Dame=1817, Ohio=1899,
                       Oklahoma=1060, Oregon=1148, Pittsburgh=2084,
                       Stanford= 732, Texas_AM=1095, Wisconsin=1725,
                       Yale=2524),
    Brigham_Young=dict(Brown=2363, Colorado= 481, Duke=2129,
                       Florida_State =2030, Louisiana =1641,
                       Louisville=1594, Michigan=1638, New_Mexico= 557,
                       North_Dakota=1214, Notre_Dame=1492, Ohio=1710,
                       Oklahoma=1126, Oregon= 825, Pittsburgh=1861,
                       Stanford= 811, Texas_AM=1195, Wisconsin=1375,
                       Yale=2262),
    Brown=dict(Colorado=1965, Duke= 669, Florida_State =1274,
               Louisiana =1541, Louisville= 920, Michigan= 744,
               New_Mexico=2172, North_Dakota=1623, Notre_Dame= 875,
               Ohio= 720, Oklahoma=1595, Oregon=3085,
               Pittsburgh= 543, Stanford=3113, Texas_AM=1734,
               Wisconsin=1111, Yale= 103),
    Colorado=dict(Duke=1667, Florida_State =1605, Louisiana =1194,
                  Louisville=1132, Michigan=1242, New_Mexico= 431,
                  North_Dakota= 963, Notre_Dame=1096, Ohio=1280,
                  Oklahoma= 664, Oregon=1249, Pittsburgh=1464,
                  Stanford=1276, Texas_AM= 799, Wisconsin= 979,
                  Yale=1866),
    Duke=dict(Florida_State = 621, Louisiana = 906, Louisville= 541,
              Michigan= 643, New_Mexico=1733, North_Dakota=1504,
              Notre_Dame= 733, Ohio= 459, Oklahoma=1187,
              Oregon=2880, Pittsburgh= 479, Stanford=2791,
              Texas_AM=1169, Wisconsin= 932, Yale= 566),
    Florida_State =dict(Louisiana = 443, Louisville= 662, Michigan= 978,
                        New_Mexico=1482, North_Dakota=1669, Notre_Dame= 925,
                        Ohio= 839, Oklahoma=1007, Oregon=2855,
                        Pittsburgh= 929, Stanford=2541, Texas_AM= 843,
                        Wisconsin=1107, Yale=1172),
    Louisiana =dict(Louisville= 754, Michigan=1106, New_Mexico=1074,
                    North_Dakota=1447, Notre_Dame= 976, Ohio= 968,
                    Oklahoma= 638, Oregon=2477, Pittsburgh=1148,
                    Stanford=2132, Texas_AM= 435, Wisconsin=1027,
                    Yale=1442),
    Louisville=dict(Michigan= 347, New_Mexico=1293, North_Dakota=1015,
                    Notre_Dame= 261, Ohio= 209, Oklahoma= 724,
                    Oregon=2319, Pittsburgh= 389, Stanford=2346,
                    Texas_AM= 841, Wisconsin= 443, Yale= 818),
    Michigan=dict(New_Mexico=1511, North_Dakota= 961, Notre_Dame= 170,
                  Ohio= 183, Oklahoma= 970, Oregon=2361,
                  Pittsburgh= 287, Stanford=2389, Texas_AM=1124,
                  Wisconsin= 388, Yale= 688),
    New_Mexico=dict(North_Dakota=1318, Notre_Dame=1363, Ohio=1447,
                    Oklahoma= 585, Oregon=1378, Pittsburgh=1631,
                    Stanford=1063, Texas_AM= 641, Wisconsin=1275,
                    Yale=2071),
    North_Dakota=dict(Notre_Dame= 813, Ohio=1078, Oklahoma= 918,
                      Oregon=1571, Pittsburgh=1182, Stanford=1882,
                      Texas_AM=1147, Wisconsin= 582, Yale=1583),
    Notre_Dame=dict(Ohio= 271, Oklahoma= 826, Oregon=2215,
                    Pittsburgh= 373, Stanford=2242, Texas_AM= 979,
                    Wisconsin= 241, Yale= 774),
    Ohio=dict(Oklahoma= 882, Oregon=2453, Pittsburgh= 192,
              Stanford=2408, Texas_AM=1049, Wisconsin= 504,
              Yale= 621),
    Oklahoma=dict(Oregon=1891, Pittsburgh=1066, Stanford=1667,
                  Texas_AM= 251, Wisconsin= 798, Yale=1495),
    Oregon=dict(Pittsburgh=2583, Stanford= 559, Texas_AM=2042,
                Wisconsin=2108, Yale=2984),
    Pittsburgh=dict(Stanford=2612, Texas_AM=1220, Wisconsin= 611,
              Yale= 442),
    Stanford=dict(Texas_AM=1701, Wisconsin=2125, Yale=3012),
    Texas_AM=dict(Wisconsin= 977, Yale=1634),
    Wisconsin=dict(Yale=1011)
    )

univ_map = Graph(univ_roads, False)



test_bases = ['A', 'B', 'C', 'D', 'E']

test_roads = dict(
    A=dict(B=6, C=20, D=5, E=14),
    B=dict(C=9, D=4, E=12),
    C=dict(D=13, E=7),
    D=dict(E=10)
)

test_map = Graph(test_roads, False)
