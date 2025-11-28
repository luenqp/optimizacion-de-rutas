import numpy as np
import random
from math import sqrt

def generar_datos(m_camiones=5, m_clientes=4, min_demanda=0.1, max_demanda=3.0, capacidad_fija=10.0, costo_fijo_x_camion=2.0, semilla=42):
    random.seed(semilla)
    
    if m_clientes > m_camiones:
        raise "No debe haber más clientes que camiones."

    I = [i for i in range(1, m_camiones + 1)]  # camiones
    J = [i for i in range(1, m_clientes + 1)]  # clientes
    
    # Demandas aleatorias
    d = np.round(np.random.uniform(min_demanda, max_demanda, size=len(J)), 1)
    
    # Capacidades de camiones
    Q = np.array([capacidad_fija for i in I])
    
    # Costos fijos de camiones
    F = np.array([costo_fijo_x_camion for i in I])

    # Matriz de costos de rutas (proporcional a las distancias)
    costo_rutas, coords = generar_dist_matrix(J, seed=semilla)

    return I, J, d, Q, F, costo_rutas

# ============================================================================
# MATRIZ DE DISTANCIAS
# ============================================================================

def generar_dist_matrix(clientes, seed=42):
    """Genera matriz de distancias basada en coordenadas Euclidianas"""
    random.seed(seed)
    
    # Coordenadas aleatorias
    coords = {0: (50, 50)}
    for k in clientes:
        coords[k] = (random.uniform(10, 90), random.uniform(10, 90))
    
    # Calcular distancias
    ubicaciones = [0] + clientes
    dist_matrix = {}
    
    for loc1 in ubicaciones:
        dist_matrix[loc1] = {}
        for loc2 in ubicaciones:
            if loc1 == loc2:
                dist_matrix[loc1][loc2] = 0.0
            else:
                x1, y1 = coords[loc1]
                x2, y2 = coords[loc2]
                distancia = sqrt((x2 - x1)**2 + (y2 - y1)**2)
                dist_matrix[loc1][loc2] = round(distancia, 2)
    
    return dist_matrix, coords
