#Modélisation des instances
import numpy as np
import math

# ----------------------------
# Classe pour une ville
# ----------------------------
class City : 
    def __init__(self,idx,x,y,deliveries=None, pickups=None):
        self.idx = idx
        self.x = x
        self.y = y
        self.deliveries = deliveries if deliveries is not None else []
        self.pickups = pickups if pickups is not None else []



# ----------------------------
# Classe pour une instance PD-TSP
# ----------------------------

class PDInstance:
    def __init__(self,cities,W0, Wmax, a=1.0, b=0.0):
        self.cities = cities
        self.n_cities = len(cities)
        self.W0 = W0
        self.Wmax = Wmax
        self.a = a
        self.b = b
        self.distances = self.compute_distances()


    def compute_distances(self):
        n = self.n_cities
        d = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                dx = self.cities[i].x - self.cities[j].x
                dy = self.cities[i].y - self.cities[j].x
                d[i,j] = math.sqrt(dx**2 + dy**2)
        
        return d
    


    def compute_weights(self, tour, pickup_plan):
        weights = []
        W = self.W0

        for city_idx in tour:
            city = self.cities[city_idx]

            #livraisons
            for w in city.deliveries:
                W -= w

            #ramassages
            for take, (w,_) in zip(pickup_plan[city_idx], city.pickups):
                if take:
                    W += w

            if W > self.Wmax:
                raise ValueError("Capacité dépassé")
            
            weights.append(W)
    
        return weights
    

    #Coût de déplacement
    def travel_cost(self,weight):
        return self.a*weight + self.b*weight*weight #si cas linéaire b = 0


    #Génération d'instances aléatoires
    @staticmethod
    def generate_random(n_cities=10, max_deliveries=2, max_pickups=2,
                        W0=10, Wmax=25, a=1.0, b=0.0):
        
        cities = []

        for i in range(n_cities):
            x,y = np.random.rand(), np.random.rand()

            deliveries = [np.random.randint(1,4) for _ in range(np.random.randint(0,max_deliveries+1))]

            pickups = [(np.random.randint(1,4) , np.random.randint(5,15)) for _ in range(np.random.randint(0,max_pickups+1))]

            cities.append(City(i,x,y,deliveries,pickups))

        return PDInstance(cities, W0, Wmax, a, b)



# ----------------------------
# Fonction globale d'évaluation d'une solution
# ----------------------------
def evaluate_solution(instance, tour, pickup_plan):
    """
    Calcule la valeur objective Z = profit_total - coût_total
    - instance : objet PDInstance
    - tour : liste d'indices des villes
    - pickup_plan : dict {city_idx: [0|1, 0|1, ...]} pour les ramassages
    """
    profit = 0.0
    cost = 0.0
    weights = instance.compute_weights(tour, pickup_plan)

    for i in range(len(tour)):
        city = instance.cities[tour[i]]

        #profits de ramassage
        for take, (_,p) in zip(pickup_plan[city.idx], city.pickups):
            if take:
                profit += p

        #coût de déplacement
        j = tour[(i+1)%len(tour)]
        d = instance.distances[city.idx, j]
        cost += d * instance.travel_cost(weights[i])

    return profit - cost 




    
    

    

