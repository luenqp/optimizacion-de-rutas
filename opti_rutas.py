import time
import pyomo.environ as pyo
from itertools import combinations

def run_optimization(I, J, F, Q, d, n_camiones, m_clientes, costo_fijo, capacidad, dist_matrix, seed):
    t0 = time.time()
    
    # Inicialización
    current_columns = []
    for i in I:
        for k in J:
            S = [k]
            current_columns.append((i, S))
    print("\nCOLUMNAS ACTUALES: ", current_columns)
    # Bucle de Generacion de Columnas
    solver = pyo.SolverFactory('appsi_highs')
    it = 1
    max_iter = 20
    iterations = []
        
    while it <= max_iter:
        print(f"ITERACIÓN {it}")
        # Resolver RMP
        rmp = build_rmp(I, J, F, Q, d, dist_matrix, current_columns, binary=False)
        res = solver.solve(rmp, tee=False)
            
        # Extraer duales
        beta = {i: rmp.dual[rmp.slot[i]] for i in I}
        alpha = {i: rmp.dual[rmp.cap[i]] for i in I}
        pi = {k: rmp.dual[rmp.cover[k]] for k in J}
            
        zLP = pyo.value(rmp.OBJ)
            
        # Pricing
        #curr_set = set(current_columns)
        rc_list = pricing_dual_rc(beta, alpha, pi, Q, dist_matrix, current_columns, I, J, d, F)
        # Contar columnas negativas
        #print(rc_list)
        nneg = sum(1 for rc, _, _ in rc_list if rc < -1e-9)
        
        iterations.append({
            'iter': it,
            'zLP': zLP,
            'negativeRC': nneg,
            'columnsAdded': nneg,
            'totalColumns': len(current_columns)
        })
            
        if nneg == 0:
            break
            
        # Agregar columnas con rc < 0
        for rc, i, S_tuple in rc_list:
            if rc < -1e-9:
                S = frozenset(S_tuple)
                current_columns.append((i, S_tuple))
            
        it += 1
            
        if it > 100:  # Límite de seguridad
            break
        
    # Resolver MIP final
    mip = build_rmp(I, J, F, Q, d, dist_matrix, current_columns, binary=True)
    res_mip = solver.solve(mip, tee=False)
    
    # Extraer solución
    z_optimal = pyo.value(mip.OBJ)
    routes = []
        
    for c in mip.COL:
        if pyo.value(mip.x[c]) > 0.5:
            i = mip.col_i[c]
            S = tuple(sorted(mip.col_S[c]))
            carga = mip.col_load[c]
            costo = mip.col_cost[c]
            routes.append({
                'Camión': i,
                'Ruta': S,
                'Carga': carga,
                'Costo': costo
            })
        
    tiempo_duracion = (time.time() - t0) / 60
    
    return {
        'z_optimal': z_optimal,
        'routes': routes,
        'iterations': iterations,
        'time_minutes': tiempo_duracion,
        'data': {
            'I': I,
            'J': J,
            'F': F,
            'Q': Q,
            'd': d,
            'n_routes': len(dist_matrix)
        }
    }

# ============================================================================
# CONSTRUCCIÓN DEL RMP
# ============================================================================

def build_rmp(I, J, F, Q, d, dist_matrix, current_columns, binary=False):
    """
    Construye el Restricted Master Problem.
    current_columns: lista de (camión, conjunto_clientes)
    """
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=I)
    m.K = pyo.Set(initialize=J)
    m.COL = pyo.Set(initialize=range(len(current_columns)))
    
    # Atributos por columna
    m.col_i = {c: current_columns[c][0] for c in m.COL}
    m.col_S = {c: current_columns[c][1] for c in m.COL}
    m.col_cost = {c: column_cost_from_route(m.col_i[c], m.col_S[c], dist_matrix, F) 
                  for c in m.COL}
    m.col_load = {c: route_load(m.col_S[c], d) for c in m.COL}
    m.col_cover = {(c, k): covers_k(m.col_S[c], k) for c in m.COL for k in m.K}
    
    # Variables
    if binary:
        m.x = pyo.Var(m.COL, domain=pyo.Binary)
    else:
        m.x = pyo.Var(m.COL, domain=pyo.NonNegativeReals)
    
    # Objetivo
    def obj_rule(mm):
        return sum(mm.col_cost[c] * mm.x[c] for c in mm.COL)
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    # Restricciones
    def slot_rule(mm, i):
        return sum(mm.x[c] for c in mm.COL if mm.col_i[c] == i) <= 1.0
    m.slot = pyo.Constraint(m.I, rule=slot_rule)

    def cap_rule(mm, i):
        return sum(mm.col_load[c] * mm.x[c] for c in mm.COL if mm.col_i[c] == i) <= Q[i-1]
    for i in m.I: print(i, '\n')
    m.cap = pyo.Constraint(m.I, rule=cap_rule)
    
    def cover_rule(mm, k):
        return sum(mm.col_cover[c, k] * mm.x[c] for c in mm.COL) == 1.0
    m.cover = pyo.Constraint(m.K, rule=cover_rule)
    
    # Duales (solo LP)
    if not binary:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    
    return m

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def fmt_float(x, nd=3):
    return f"{float(x):.{nd}f}"

def route_load(S, d) -> float:
    return sum(d[k-1] for k in S)

def column_cost_from_route(i, S, dist_matrix, F) -> float:
    """Calcula el costo de una ruta: F_i + distancia_ruta"""
    if len(S) == 0:
        return F[i-1]
    
    S_list = list(S)
    if len(S_list) == 1:
        # Ruta unitaria: depot → k → depot
        costo_ruta = 2 * dist_matrix[0, S_list[0]]
    else:
        # Ruta múltiple: depot → k1 → k2 → ... → depot
        costo_ruta = dist_matrix[0, S_list[0]]
        for j in range(len(S_list) - 1):
            costo_ruta += dist_matrix[S_list[j], S_list[j+1]]
        costo_ruta += dist_matrix[S_list[-1], 0]
    
    return F[i-1] + costo_ruta

def covers_k(S, k) -> int:
    return 1 if k in S else 0

# ============================================================================
# PRICING CON SUBPROBLEMA DE OPTIMIZACIÓN
# ============================================================================

def pricing_subproblem_best_route(i, beta_i, alpha_i, pi, Q_i, d, dist_matrix, 
                                   current_columns_set, J, F):
    """
    Encuentra la mejor ruta con rc < 0 para el camión i usando búsqueda exhaustiva
    optimizada con programación dinámica simplificada.
    
    Minimiza: rc = (F_i + costo_ruta) - beta_i - alpha_i * carga - sum(pi_k for k in S)
    """
    mejor_ruta = None
    mejor_rc = 0
    
    # Estados: (nodo_actual, clientes_visitados, carga_actual)
    # Valor: (costo_acumulado, secuencia)
    
    # Generar todas las combinaciones factibles de clientes
    for r in range(1, len(J) + 1):
        for S_tuple in combinations(J, r):
            print("\n\nS_TUPLE:\n", S_tuple)
            # Verificar capacidad
            carga = sum(d[k-1] for k in S_tuple)
            if carga > Q_i:
                continue
            
            S = frozenset(S_tuple)
            
            # Verificar si ya está en columnas actuales
            if (i, S) in current_columns_set:
                continue
            
            # Calcular costo de la ruta (aproximación: orden dado)
            if len(S_tuple) == 1:
                costo_ruta = 2 * dist_matrix[0, S_tuple[0]]
            else:
                costo_ruta = dist_matrix[0, S_tuple[0]]
                for j in range(len(S_tuple) - 1):
                    costo_ruta += dist_matrix[S_tuple[j], S_tuple[j+1]]
                costo_ruta += dist_matrix[S_tuple[-1], 0]
            
            # Calcular coste reducido
            rc = (F[i-1] + costo_ruta) - beta_i - alpha_i * carga - sum(pi[k] for k in S)
            
            if rc < mejor_rc:
                mejor_rc = rc
                mejor_ruta = S
    
    return mejor_ruta, mejor_rc

def pricing_dual_rc(beta, alpha, pi, Q, dist_matrix, current_columns_set, I, J, d, F):
    """
    Ejecuta pricing para todos los camiones y retorna todas las columnas
    con rc < 0 encontradas.
    """
    nuevas_columnas = []
    
    for i in I:
        ruta, rc = pricing_subproblem_best_route(
            i, beta[i], alpha[i], pi, Q[i-1], d, dist_matrix, current_columns_set, J, F)
        
        if ruta is not None:
            nuevas_columnas.append((rc, i, ruta))
    
    # Ordenar por rc (más negativo primero)
    nuevas_columnas.sort(key=lambda x: x[0])
    
    return nuevas_columnas




