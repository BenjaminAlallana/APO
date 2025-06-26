import math, random
from scipy.special import  expit
import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np


class Problem:
    def __init__(self):
        self.n_clients = 12
        self.n_hubs = 5

        # Distancias cliente-hub (12 x 5)
        self.distancias = [
            [6, 9, 12, 7, 8],
            [5, 8, 6, 10, 9],
            [8, 7, 9, 6, 11],
            [7, 5, 6, 9, 10],
            [9, 6, 8, 12, 7],
            [6, 5, 9, 11, 8],
            [8, 7, 5, 6, 10],
            [7, 6, 9, 8, 6],
            [5, 9, 10, 7, 11],
            [9, 6, 7, 8, 9],
            [6, 10, 8, 7, 5],
            [8, 6, 9, 10, 6]
        ]

        # Costo fijo por abrir cada hub
        self.costs = [20, 25, 18, 22, 19]

        # Capacidad máxima por hub (total debe cubrir 12 clientes)
        self.capacidad = [3, 3, 3, 3, 3]  # total 15 ≥ 12

        # Distancia máxima tolerada
        self.D_max = 9

    def check(self, x):
        # Derivar hubs desde asignaciones
        hubs = []
        for _ in range(self.n_hubs):
            hubs.append(0)
        for h in x:
            hubs[h] = 1

        # Verificar capacidad y distancia máxima
        conteo = [0] * self.n_hubs
        for c in range(self.n_clients):
            h = x[c]

            # Verificar distancia máxima
            if self.distancias[c][h] > self.D_max:
                return False

            # Hub debe estar activo (ya se cumple por construcción)
            conteo[h] += 1
            if conteo[h] > self.capacidad[h]:
                return False

        return True

    def fit(self, x):
        # Derivar hubs desde asignaciones
        hubs = []
        for _ in range(self.n_hubs):
            hubs.append(0)

        for h in x:
            hubs[h] = 1

        # Calcular distancia total de asignación
        total = 0
        for c in range(self.n_clients):
            total += self.distancias[c][x[c]]

        # Agregar costo fijo de hubs activados
        for j in range(self.n_hubs):
            if hubs[j] == 1:
                total += self.costs[j]

        return total

    def keep_domain(self, v):
        v = max(min(v,60),-60)
        probs = []
        for _ in range(self.n_hubs):
            probs.append(expit(v + random.gauss(0, 0.5)))

        total = sum(probs)
        normalized = [p / total for p in probs]

        # Muestreo proporcional
        r = random.random()
        acc = 0
        for i, p in enumerate(normalized):
            acc += p
            if r <= acc:
                return i
        return self.n_hubs - 1


class Puffin:
    def __init__(self):
        self.p = Problem()
        self.dimension = self.p.n_clients
        self.position_X = []
        self.position_Y = []
        self.position_Z = []
        self.position_W = []
        self.position_P = []
        self.p_best = []

        for j in range(self.dimension):
            self.position_X.append(random.randint(0,self.p.n_hubs -1))
            self.position_Y.append(0)
            self.position_Z.append(0)
            self.position_W.append(0)

    def fitness(self) -> int:
        return self.p.fit(self.position_X)

    def fitness_p_best(self) -> int:
        return self.p.fit(self.p_best)

    def is_feasible(self) -> bool:
        return self.p.check(self.position_X)

    def is_better_than_p_best(self) -> bool:
        return self.fitness() < self.fitness_p_best()

    def is_better_than(self, g) -> bool:
        return self.fitness() < g.fitness_p_best()

    def update_p_best(self):
        self.p_best = self.position_X.copy()

    #flight Levy
    def levy(self, beta=1.5):
      sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
      u = random.gauss(0, sigma_u)
      v = random.gauss(0, 1)
      step = u / abs(v) ** (1 / beta)
      return step

    #Fase Aera
    def airSearch(self):
      for j in range(self.dimension):
        R = round( 0.5 * (0.05 + random.random()) * random.gauss(0,1))
        r = random.randint(1, self.dimension -1)
        self.position_Y[j] = self.position_X[j] + (self.position_X[j] - self.position_X[r]) * self.levy() + R
        self.position_Y[j] = self.p.keep_domain(self.position_Y[j])

    def divePredation(self):
      for j in range(self.dimension):
        self.position_Z[j] = self.position_Y[j] * math.tan((random.random() - 0.05) * math.pi)
        self.position_Z[j] = self.p.keep_domain(self.position_Z[j])



    #Fase Acuatica
    def gatherForFood(self, F):
      for j in range(self.dimension):
        opciones = [h for h in range(0, self.dimension) if h != j]
        r1, r2, r3 = random.sample(opciones, 3)
        if (random.random() >= 0.5):
          self.position_W[j] = self.position_X[r1] + F * (self.position_X[r2] - self.position_X[r3]) * self.levy()
        else:
          self.position_W[j] = self.position_X[r1] + F * (self.position_X[r2] - self.position_X[r3])
        self.position_W[j] = self.p.keep_domain(self.position_W[j])

    def strengthenSearch(self,T,t):
      for j in range(self.dimension):
        f = 0.1 * (random.random() -1) * ((T-t)/T)
        self.position_Y[j] = self.position_W[j] * (1 + f)
        self.position_Y[j] = self.p.keep_domain(self.position_Y[j])

    def avoidPredators(self, F):
      for j in range(self.dimension):
        opciones = [h for h in range(0, self.dimension) if h != j]
        r1, r2 = random.sample(opciones, 2)
        if (random.random() >= 0.5):
          self.position_Z[j] = self.position_X[j] + F * (self.position_X[r1] - self.position_X[r2]) * self.levy()
        else:
          self.position_Z[j] = self.position_X[j] + F * (self.position_X[r1] - self.position_X[r2])
        self.position_Z[j] = self.p.keep_domain(self.position_Z[j])

    #QuickSort para posicion P
    def quicksort(self, arr):
        if len(arr) <= 1:
            return arr
        else:
            pivot = arr[0]
            pivot_fit = self.p.fit(pivot)
            left = [x for x in arr[1:] if self.p.fit(x) > pivot_fit]
            right = [x for x in arr[1:] if self.p.fit(x) <= pivot_fit]
            return self.quicksort(left) + [pivot] + self.quicksort(right)

    #Combinacion de soluciones
    def fusion_acuatica(self):
        self.position_P = []
        self.position_P.append(self.position_W.copy())
        self.position_P.append(self.position_Y.copy())
        self.position_P.append(self.position_Z.copy())
        self.position_P = self.quicksort(self.position_P)
        self.position_X = self.position_P[0]

    def airFusion(self):
        self.position_P = []
        self.position_P.append(self.position_Y.copy())
        self.position_P.append(self.position_Z.copy())
        self.position_P = self.quicksort(self.position_P)
        self.position_X = self.position_P[0]

    def copy(self, other):
        if isinstance(other, Puffin):
          self.position_X = other.position_X.copy()
          self.position_Y = other.position_Y.copy()
          self.position_Z = other.position_Z.copy()
          self.position_W = other.position_W.copy()
          self.position_P = other.position_P.copy()
          self.p_best = other.p_best.copy()

    def __str__(self):
        return f"p_best: {self.p_best}, fitness: {self.fitness_p_best()}"


class APO:
    def  __init__(self):
        self.popSize = 30
        self.max_iter= 50
        self.C = 0.5
        self.F = 0.5
        self.swarm = []
        self.g = None
        self.qMetric = 0
        self.convergence = []

    def solve(self):
        self.initial()
        self.evolve()

    def initial(self):
        for i in range(self.popSize):
            feasible = False
            while not feasible:
                p = Puffin()
                feasible = p.is_feasible()
            p.update_p_best()
            self.swarm.append(p)

        self.g = Puffin()
        self.g.copy(self.swarm[0])

        for i in range(1, self.popSize):
            if (self.swarm[i].is_better_than(self.g)):
                self.g.copy(self.swarm[i])

        self.show(1)

    def evolve(self):
        t = 1
        while t <= self.max_iter:
            B = 2 * math.log( 1 / random.random() ) * ( 1 - (t / self.max_iter))
            for i in range(self.popSize):
                if( B > self.C ):
                    #Busqueda aera
                    feasible = False
                    p = Puffin()
                    while not feasible:
                        p.copy(self.swarm[i])
                        p.airSearch()
                        p.divePredation()
                        p.airFusion()
                        feasible = p.is_feasible()
                    self.swarm[i].copy(p)
                else:
                    #Busqueda Acuatica
                    feasible = False
                    p = Puffin()
                    while not feasible:
                        p.copy(self.swarm[i])
                        p.gatherForFood(self.F)
                        p.strengthenSearch(self.max_iter, t)
                        p.avoidPredators(self.F)
                        p.fusion_acuatica()
                        feasible = p.is_feasible()
                    self.swarm[i].copy(p)

                if (self.swarm[i].is_better_than_p_best()):
                    self.swarm[i].update_p_best()
                if (self.swarm[i].is_better_than(self.g)):
                    self.g.copy(self.swarm[i])

            t = t + 1
            self.show(t)
        self.qmetric()

    def show(self, t):
        fitness = self.g.fitness_p_best()
        self.convergence.append(fitness)
        print(f"t = {t}, g = {self.g}")


    def qmetric(self):
        q = (190 - self.g.fitness_p_best()) / (190 - 151)
        self.qMetric = 2 **(q ** ( 10 ** (3))) - 1

    def plot_convergence(self, instancia_id=None):
        plt.figure()
        plt.plot(self.convergence, label='Mejor fitness')
        plt.xlabel('Iteración')
        plt.ylabel('Fitness')
        plt.title(f'Convergencia APO{" - Instancia " + str(instancia_id) if instancia_id else ""}')
        plt.grid(True)
        plt.legend()
        filename = f"convergencia_{instancia_id}.png" if instancia_id else "convergencia.png"
        plt.savefig(filename)
        plt.close()



experimentos = 30
resultados = [["fitness", "QMetric"]]
for i in range(1,experimentos + 1):
    print(f'Instancia: {i}')
    apo = APO()
    apo.solve()
    resultados.append([apo.g.fitness_p_best(), apo.qMetric])
    print(f'Mejor Resultado: {apo.g.fitness_p_best()}, QMetric: {apo.qMetric}')
    apo.plot_convergence(i)

# Escribir en un archivo CSV
with open('instancia_media.csv', 'w', newline='', encoding='utf-8') as archivo:
    escritor = csv.writer(archivo)
    escritor.writerows(resultados)


def estadisticas_centrales(resultados, filename="estadisticas.csv"):
    fitness_vals = [fila[0] for fila in resultados[1:]]  # Excluye encabezado
    qmetrics_vals = [fila[1] for fila in resultados[1:]]

    stats = [
        ["Métrica", "Media", "Mediana", "Desviación Estándar", "Mínimo", "Máximo", "IQR"],
        [
            "Fitness",
            round(statistics.mean(fitness_vals), 2),
            round(statistics.median(fitness_vals), 2),
            round(statistics.stdev(fitness_vals), 2),
            min(fitness_vals),
            max(fitness_vals),
            round(np.percentile(fitness_vals, 75) - np.percentile(fitness_vals, 25), 2)
        ],
        [
            "QMetric",
            round(statistics.mean(qmetrics_vals), 4),
            round(statistics.median(qmetrics_vals), 4),
            round(statistics.stdev(qmetrics_vals), 4),
            round(min(qmetrics_vals), 4),
            round(max(qmetrics_vals), 4),
            round(np.percentile(qmetrics_vals, 75) - np.percentile(qmetrics_vals, 25), 4)
        ]
    ]

    # Escribir en CSV
    with open(filename, 'w', newline='', encoding='utf-8') as archivo:
        writer = csv.writer(archivo)
        writer.writerows(stats)

    print(f"\n📁 Estadísticas guardadas en '{filename}'")

estadisticas_centrales(resultados)
