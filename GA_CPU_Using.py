# ga_knapsack_vectorized.py
import time
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

def true(prob):
    return rng.random() < prob

class Population:
    """Vectorized population using numpy arrays (shape: (N, L), dtype=np.int8 or bool)."""

    def __init__(self, pop_array):
        # pop_array: numpy array shape (N, L)
        self._pop = np.asarray(pop_array, dtype=np.int8)
        self._fitness = None

    @property
    def pop(self):
        return self._pop

    @property
    def fitness(self):
        return self._fitness

    def evaluation(self, evaluator):
        """
        evaluator must accept either:
          - a single individual (1D array) -> returns scalar fitness (backward-compatible), or
          - a 2D numpy array (N, L) -> return 1D fitness array (vectorized).
        KnapsackProblem provides a vectorized fitness implementation, so we use it.
        """
        pop = self._pop
        # try vectorized call first
        try:
            fits = evaluator(pop)
            # if result is scalar or list-like, convert to numpy array
            fits = np.asarray(fits, dtype=float)
            if fits.ndim == 0:
                # single scalar (weird), fallback to loop
                raise TypeError
            if fits.shape[0] != pop.shape[0]:
                raise ValueError("Vectorized evaluator returned unexpected shape.")
            self._fitness = fits
            return self
        except Exception:
            # fallback to per-individual evaluation (slow; kept for compatibility)
            n = pop.shape[0]
            fits = np.empty(n, dtype=float)
            for i in range(n):
                fits[i] = evaluator(pop[i])
            self._fitness = fits
            return self

    def selection(self, selection_mechanism):
        if selection_mechanism == "TMS":
            return self.selection_TMS()
        elif selection_mechanism == "RWS":
            return self.selection_RWS()
        else:
            raise ValueError("Unknown selection mechanism")

    def selection_TMS(self):
        """Tournament selection (pairwise): for each index i, pick random j and take the fitter."""
        assert self._fitness is not None
        pop = self._pop
        fitness = self._fitness
        n = pop.shape[0]
        opponents = rng.integers(0, n, size=n)
        # mask True where i is better than opponent
        mask = fitness > fitness[opponents]
        # indices chosen: where mask True -> i, else -> opponent
        indices = np.where(mask, np.arange(n), opponents)
        # select and copy to avoid aliasing
        self._pop = pop[indices].copy()
        self._fitness = None
        return self

    def selection_RWS(self):
        """Roulette Wheel Selection with handling of zero or constant fitness."""
        assert self._fitness is not None
        pop = self._pop
        fitness = np.array(self._fitness, dtype=float)
        n = pop.shape[0]

        if np.ptp(fitness) == 0 or np.max(fitness) <= 0:
            # uniform selection if all equal or non-positive
            idxs = rng.integers(0, n, size=n)
            self._pop = pop[idxs].copy()
        else:
            # shift negatives/zeros: keep zeros as zero, shift positive by min positive
            pos = fitness > 0
            minpos = np.min(fitness[pos]) if np.any(pos) else 0.0
            norm_f = np.where(fitness == 0, 0.0, fitness - minpos)
            total = np.sum(norm_f)
            if total <= 0:
                idxs = rng.integers(0, n, size=n)
            else:
                probs = norm_f / total
                # rng.choice on indices
                idxs = rng.choice(n, size=n, replace=True, p=probs)
            self._pop = pop[idxs].copy()

        self._fitness = None
        return self

    def three_point_crossover(self, prob):
        """Three-point crossover (keeps previous semantics). Operates pairwise."""
        pop = self._pop
        n, L = pop.shape
        idx = rng.permutation(n)
        half = n // 2
        for i in range(half):
            a_idx = idx[i]
            b_idx = idx[half + i]
            a = pop[a_idx]
            b = pop[b_idx]
            if true(prob):
                poss = rng.choice(L - 1, size=3, replace=False)
                poss.sort()
                s1, e1 = poss[0] + 1, poss[1] + 1
                s2, e2 = poss[2] + 1, L
                # swap slices
                tmp = a[s1:e1].copy()
                a[s1:e1] = b[s1:e1]
                b[s1:e1] = tmp
                tmp = a[s2:e2].copy()
                a[s2:e2] = b[s2:e2]
                b[s2:e2] = tmp
        return self

    def uniform_crossover(self, prob, gene_prob=0.5):
        """Vectorized-ish uniform crossover: for each pair, create a boolean mask of genes to swap."""
        pop = self._pop
        n, L = pop.shape
        idx = rng.permutation(n)
        half = n // 2

        for i in range(half):
            a_idx = idx[i]
            b_idx = idx[half + i]
            a = pop[a_idx]
            b = pop[b_idx]
            if true(prob):
                # gene mask vectorized for this pair
                mask = rng.random(L) < gene_prob
                if np.any(mask):
                    # swap masked positions
                    tmp = a[mask].copy()
                    a[mask] = b[mask]
                    b[mask] = tmp
        return self
    
    def crossover_gender_group(
        self, prob=0.9, gene_prob=0.5,
        tournament_k=60, partners=30
    ):
        """
        Gender-split + tournament selection + multi-partner uniform crossover.

        새로운 GA 구조에 맞춘 버전:
        - self._pop: shape (N, L) numpy array
        - self._fitness: numpy array (N,)
        """
        pop = self._pop
        fitness = self._fitness
        n, L = pop.shape
        half = n // 2

        males = np.arange(0, half)
        females = np.arange(half, n)

        rng.shuffle(males)
        rng.shuffle(females)

        # -----------------------
        # (1) Tournament selection: male champion
        # -----------------------
        m_candidates = rng.choice(males, size=tournament_k, replace=False)
        best_male_idx = m_candidates[np.argmax(fitness[m_candidates])]

        # -----------------------
        # (2) Tournament selection: female champion
        # -----------------------
        f_candidates = rng.choice(females, size=tournament_k, replace=False)
        best_female_idx = f_candidates[np.argmax(fitness[f_candidates])]

        # -----------------------
        # (3) Best male ↔ random females crossover
        # -----------------------
        chosen_females = rng.choice(females, size=partners, replace=False)

        for f_idx in chosen_females:
            if true(prob):
                mask = rng.random(L) < gene_prob
                if mask.any():
                    # numpy array 상호 스왑
                    tmp = pop[best_male_idx, mask].copy()
                    pop[best_male_idx, mask] = pop[f_idx, mask]
                    pop[f_idx, mask] = tmp

        # -----------------------
        # (4) Best female ↔ random males crossover
        # -----------------------
        chosen_males = rng.choice(males, size=partners, replace=False)

        for m_idx in chosen_males:
            if true(prob):
                mask = rng.random(L) < gene_prob
                if mask.any():
                    tmp = pop[best_female_idx, mask].copy()
                    pop[best_female_idx, mask] = pop[m_idx, mask]
                    pop[m_idx, mask] = tmp

        return self


    def mutation(self, prob):
        """Vectorized bit-flip mutation over the whole population."""
        pop = self._pop
        mask = rng.random(pop.shape) < prob
        # flip bits where mask True
        pop[mask] = 1 - pop[mask]
        self._pop = pop
        return self

    @staticmethod
    def encode(individual):
        # accept numpy 1D array
        return "".join("1" if int(b) else "0" for b in individual)

    def dump(self):
        fits = self._fitness
        if fits is None:
            fits = [None] * len(self._pop)
        string = ""
        for ind, fit in zip(self._pop, fits):
            code = self.encode(ind)
            if fit is None:
                string += f"{code},None\n"
            else:
                string += f"{code},{fit:.6f}\n"
        return string

class KnapsackProblem:
    """Vectorized Knapsack Problem."""

    def __init__(self, filename=None):
        self.pop_size = 200
        self.max_generations = 200
        self.Pc = 0.9
        self.Pm = 0.01
        self.weights = None
        self.profits = None
        self.capacity = 0.0
        self.len = 0
        self.log = {}
        if filename:
            self.read_data(filename)

    def read_data(self, filename):
        with open(filename, "r") as f:
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
            weights = []
            profits = []
            for line in f:
                i, w, p = line.strip().split()
                weights.append(float(w))
                profits.append(float(p))
            self.weights = np.array(weights, dtype=float)
            self.profits = np.array(profits, dtype=float)
            self.len = len(weights)
            self.log = {}
        return self

    def fitness_function(self, individual):
        """
        Supports both:
          - vectorized call: individual is 2D numpy array -> returns 1D fitness array
          - scalar call: individual is 1D (fallback)
        Fitness: sum_profit if sum_weight <= capacity else 0
        """
        if isinstance(individual, np.ndarray) and individual.ndim == 2:
            # vectorized evaluation
            # individuals are ints 0/1
            sum_weights = individual @ self.weights      # shape (N,)
            sum_profits = individual @ self.profits      # shape (N,)
            # apply capacity constraint
            fitness = np.where(sum_weights <= self.capacity, sum_profits, 0.0)
            return fitness
        else:
            # fallback single individual
            total_w = float(np.dot(individual, self.weights))
            total_p = float(np.dot(individual, self.profits))
            return total_p if total_w <= self.capacity else 0.0

    def initialize_pop(self):
        # return numpy array (pop_size, len)
        return rng.integers(0, 2, size=(self.pop_size, self.len), dtype=np.int8)

    def solve_RWS(self, pop=None):
        if pop is None:
            pop = self.initialize_pop()
        generation = Population(pop)

        fitavg = []
        fitmax = []

        for gen in range(self.max_generations):
            generation.evaluation(self.fitness_function)
            fit = generation.fitness
            current_max = np.max(fit)
            fitavg.append(np.average(fit))
            fitmax.append(current_max)

            if gen % 10 == 0:
                print(f"RWS Generation {gen:3d}: Max Fitness = {current_max:.6f}")

            generation.selection("RWS")
            generation.evaluation(self.fitness_function)
            generation.crossover_gender_group(self.Pc).mutation(self.Pm)

        self.log = {"avg": fitavg, "max": fitmax, "pop": generation.pop}
        return generation.pop

    def solve_TMS(self, pop=None):
        if pop is None:
            pop = self.initialize_pop()
        generation = Population(pop)

        fitavg = []
        fitmax = []

        for gen in range(self.max_generations):
            generation.evaluation(self.fitness_function)
            fit = generation.fitness
            current_max = np.max(fit)
            fitavg.append(np.average(fit))
            fitmax.append(current_max)

            if gen % 10 == 0:
                print(f"TMS Generation {gen:3d}: Max Fitness = {current_max:.6f}")

            generation.selection("TMS")
            generation.evaluation(self.fitness_function)
            generation.crossover_gender_group(self.Pc).mutation(self.Pm)

        self.log = {"avg": fitavg, "max": fitmax, "pop": generation.pop}
        return generation.pop

def hw1(filename):
    problem = KnapsackProblem(filename)
    problem2 = KnapsackProblem(filename)

    t = time.perf_counter()
    pop = problem.initialize_pop()
    problem.solve_TMS(pop)
    logtms = problem.log
    dt = time.perf_counter() - t
    print(f"Tournament time: {dt:.6f}")
    print(f"Tournament max fitness: {max(logtms['max']):.6f}")

    t = time.perf_counter()
    pop = problem2.initialize_pop()
    problem2.solve_RWS(pop)
    logrws = problem2.log
    dt = time.perf_counter() - t
    print(f"Roulette time: {dt:.6f}")
    print(f"Roulette max fitness: {max(logrws['max']):.6f}")

    avgtms = logtms["avg"]
    avgrws = logrws["avg"]

    plt.title("0/1 Knapsack fitness value trace")
    plt.plot(range(problem.max_generations), avgtms, label="Pairwise Tournament Selection")
    plt.plot(range(problem.max_generations), avgrws, label="Roulette Wheel Selection")
    plt.legend()
    plt.savefig('result_graph.png')
    plt.close()

if __name__ == '__main__':
    t = time.perf_counter()
    hw1("Data(0-1Knapsack).txt")
    dt = time.perf_counter() - t
    print(f"Total time: {dt:.6f}")
