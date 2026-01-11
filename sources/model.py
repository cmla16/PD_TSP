#Modélisation des instances
import numpy as np
import math

class City : 
    def __init__(self,idx,x,y,deliveries=None, pickups=None):
        self.idx = idx
        self.x = x
        self.y = y
        self.deliveroes = deliveries if deliveries is not None else []
        self.pickups = pickups if pickups is not None else []



#Instance complète du PD-TSP
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
    
    
    #Fonction d'évaluation
    

    

