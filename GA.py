import time
import numpy as np


rng = np.random.default_rng(seed=42)


def true(prob):
    """True in probability of prob"""
    return rng.random() < prob


class Population:
    """Genetic Algorithm
    Population(list).evaluation(function).selection().crossover(Pc).mutation(Pm)
    """
    def __init__(self, pop):
        self._pop = pop
        self._fitness = None

    @property
    def pop(self):
        return self._pop

    @property
    def fitness(self):
        return self._fitness

    def evaluation(self, evaluator):
        """Evaluation"""
        fitness = []
        for individual in self._pop:
            val = evaluator(individual)
            fitness.append(val)
        self._fitness = fitness
        return self

    def selection(self, selection_mechanism):
        if selection_mechanism == "TMS":
            return self.selection_TMS()
        elif selection_mechanism == "RWS":
            return self.selection_RWS()
        else:
            raise ValueError

    def selection_TMS(self):
        """TournamentSelection"""
        assert(self._fitness is not None)
        number = len(self._pop)

        pop = []
        for i in range(number):
            x = rng.integers(number)
            individual = self._pop[i] if self._fitness[i] > self._fitness[x] else self._pop[x]
            pop.append(individual.copy())

        self._pop = pop
        self._fitness = None
        return self

    def selection_RWS(self):
        """RouletteWheelSelection"""
        assert(self._fitness is not None)
        fitness = np.array(self._fitness)
        # if fitness values are all the same or all individuals are bad solutions
        if np.ptp(fitness) == 0 or max(fitness) <= 0:
            # uniform selection
            self._pop = [a.copy() for a in rng.choice(self._pop, replace=True, size=len(self._pop))]
        else:
            # subtract minimum fitness to all fitness (excluding 0), to improve performance
            pos_fitness = fitness[fitness > 0]
            n_fitness = np.where(fitness==0, 0, (fitness - np.min(pos_fitness)))
            self._pop = [a.copy() for a in rng.choice(self._pop, replace=True, p=n_fitness/sum(n_fitness), size=len(self._pop))]
        self._fitness = None
        return self

    def crossover(self, prob):
        """Crossover"""
        number = len(self._pop)
        idx = rng.permutation(number)
        halfnumber = number // 2
        for i in range(halfnumber):
            a = self._pop[idx[i]]
            b = self._pop[idx[halfnumber + i]]
            length = len(a)
            if true(prob):
                poss = rng.choice(range(length-1), size=3, replace=False)
                poss.sort()
                start1, end1 = poss[0] + 1, poss[1] + 1
                start2, end2 = poss[2] + 1, length
                # three point crossover
                a[start1 : end1], b[start1 : end1] = b[start1 : end1], a[start1 : end1]
                a[start2 : end2], b[start2 : end2] = b[start2 : end2], a[start2 : end2]
        return self
    #########
    def uniform_crossover(self,prob,gene_prob=0.5):
        """ TODO: 필수 구현 """
        print("박진현: 242122") # 학생 수행 확인용입니다. 
        
        # 개체의 인덱스를 무작위 선택하기 위한 작업, crossover와 동일한 로직
        number = len(self._pop)
        idx = rng.permutation(number)
        halfnumber = number // 2

        for i in range(halfnumber):
            # 무모 개체 선택
            a = self._pop[idx[i]]
            b = self._pop[idx[halfnumber + i]]
            length = len(a)
            
            # 개체의 crossover 실행 여부
            if true(prob):
                for g in range(length):
                    
                    # 유전자의 crossover 실행 여부
                    if true(gene_prob):
                        # 부모의 유전자 교환
                        a[g], b[g] = b[g], a[g]
        return self

    #########
    def mutation(self, prob):
        """Mutation"""
        pop = []
        for ind in self._pop:
            individual = ind.copy()
            length = len(individual)
            for i in range(length):
                if true(prob):
                    individual[i] = 1 - individual[i]
            pop.append(individual)
        self._pop = pop
        return self

    @staticmethod
    def encode(individual):
        # individual must be a boolean iterable
        # convert boolean list into 01 string
        return "".join(["1" if b else "0" for b in individual])

    def dump(self):
        fits = self._fitness
        if fits is None:
            fits = [None] * len(self._pop)
        string = ""
        for ind, fit in zip(self._pop, fits):
            code = self.encode(ind)
            string += f"{code},{fit:.6f}\n"
        return string


class KnapsackProblem:
    """Define a Knapsack Problem and Solve."""

    def __init__(self, filename=None):
        self.pop_size = 100
        self.max_generations = 100
        self.Pc = 0.9
        self.Pm = 0.01
        self.weights = []
        self.profits = []
        self.capacity = 0
        self.len = 0
        self.log = {}
        if filename:
            self.read_data(filename)

    def read_data(self, filename):
        """Parse data from file."""
        with open(filename, "r") as f:
            "Total weight capacity: %d "
            # header
            table = False
            for line in f:
                iwp = line.strip().split()
                if len(iwp) >= 4 and iwp[2] == "capacity":
                    self.capacity = float(iwp[3])
                elif iwp == ["item_index", "weight", "profit"]:
                    table = True
                    break
            if not table:
                raise ValueError("table not found.")
            # body
            weights = []
            profits = []
            for line in f:
                i, w, p = line.strip().split()
                weights.append(float(w))
                profits.append(float(p))
            self.weights = weights
            self.profits = profits
            self.log = {}
            self.len = len(weights)
        return self

    def fitness_function(self, individual):
        """Calculate fitness of a code."""
        sum_weight = 0
        sum_profit = 0
        for b, w, p in zip(individual, self.weights, self.profits):
            if b:
                sum_weight += w
                sum_profit += p

        value = sum_profit if sum_weight <= self.capacity else 0
        return value

    def initialize_pop(self):
        # initialize
        pop = []
        for _ in range(self.pop_size):
            individual = rng.integers(2, size=self.len)
            pop.append(individual)
        return pop


    def solve_RWS(self, pop=None):
    
        if pop is None:
            pop = self.initialize_pop()
        generation = Population(pop)

        fitavg = []
        fitmax = []
        # while stopping condition is not fulfilled do GA
        for gen in range(self.max_generations):
            generation.evaluation(self.fitness_function)
            fit = generation._fitness
            current_max = np.max(fit)
            fitavg.append(np.average(fit))
            fitmax.append(current_max)
            
            # 10 세대마다 fitness 출력
            if gen % 10 == 0:
                print(f"RWS Generation {gen:3d}: Max Fitness = {current_max:.6f}")
                
            generation.selection("RWS").uniform_crossover(self.Pc).mutation(self.Pm)

        self.log = {"avg": fitavg, "max": fitmax, "pop": generation.pop}
        return generation.pop

    def solve_TMS(self, pop=None):
        """Solve Knapsack problem using Tournament Selection, 3 Point Crossover & Bitwise Mutation."""
        if pop is None:
            pop = self.initialize_pop()
        generation = Population(pop)

        fitavg = []
        fitmax = []
        # while stopping condition is not fulfilled do GA
        for gen in range(self.max_generations):
            generation.evaluation(self.fitness_function)
            fit = generation._fitness
            current_max = np.max(fit)
            fitavg.append(np.average(fit))
            fitmax.append(current_max)
            
            # 10 세대마다 fitness 출력
            if gen % 10 == 0:
                print(f"TMS Generation {gen:3d}: Max Fitness = {current_max:.6f}")
                
            generation.selection("TMS").uniform_crossover(self.Pc).mutation(self.Pm)

        self.log = {"avg": fitavg, "max": fitmax, "pop": generation.pop}
        return generation.pop
        
    

def hw1(filename):
    # variable filename should be the exact filename
    problem = KnapsackProblem(filename)
    problem2 = KnapsackProblem(filename)

    t = time.perf_counter()

    pop = problem.initialize_pop()
    problem.solve_TMS(pop)
    logtms = problem.log

    dt = time.perf_counter() - t
    print(f"Tournament time: {dt:.6f}")
    print(f"Tournament max fitness: {max(logtms['max']):.6f}")  # TMS의 최대 fitness 출력

    t = time.perf_counter()

    pop = problem2.initialize_pop()
    problem2.solve_RWS(pop)
    logrws = problem2.log

    dt = time.perf_counter() - t
    print(f"Roulette time: {dt:.6f}")
    print(f"Roulette max fitness: {max(logrws['max']):.6f}")   # RWS의 최대 fitness 출력

    import matplotlib.pyplot as plt
    avgtms = logtms["avg"]
    avgrws = logrws["avg"]

    plt.title("0/1 Knapsack fitness value trace")
    plt.plot(range(problem.max_generations), avgtms, label="Pairwise Tournament Selection")
    plt.plot(range(problem.max_generations), avgrws, label="Roulette Wheel Selection")
    plt.legend()
    plt.savefig('result_graph.png')  # show() 대신 파일로 저장
    plt.close()

if __name__ == '__main__':
    t = time.perf_counter()
    hw1("Data(0-1Knapsack).txt")
    dt = time.perf_counter() - t
    print(f"Total time: {dt:.6f}")  