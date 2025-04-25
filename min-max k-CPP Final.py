# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:21:44 2025

@author: Francisco FV C
"""
# --------------------------------------------------------------------------------
# Importaciones
# --------------------------------------------------------------------------------
from pulp import LpVariable, LpProblem, lpSum, LpMinimize, LpStatusOptimal, PULP_CBC_CMD

# --------------------------------------------------------------------------------
# Insertar datos del grafo
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Función de programación lineal
# --------------------------------------------------------------------------------
def ppl(datos_region, datos_deposito, K):
    # ----------------------------------------------------------------
    # 1) Recolección de arcos con servicio (regiones)
    # ----------------------------------------------------------------
    c_sencillos = [
        arco for data in datos_region.values()
        for arco in data["sencillos"]
    ]
    c_camellon  = [
        arco for data in datos_region.values()
        for arco in data["camellon"]
    ]
    c_dobles    = [
        arco for data in datos_region.values()
        for arco in data["dobles"]
    ]
    
    # Procesamiento de arcos con servicio
    a_camellon  = [(ar[1], ar[0], ar[2]) for ar in c_camellon] + c_camellon
    a_dobles    = [(ar[1], ar[0], ar[2]) for ar in c_dobles]   + c_dobles
    a_sencillos = c_sencillos
    a_completos = a_sencillos + a_camellon + a_dobles
    
    # ----------------------------------------------------------------
    # 2) Recolección de arcos sin servicio (depósito)
    # ----------------------------------------------------------------
    dc_sencillos = datos_deposito["sencillos"]
    dc_camellon  = datos_deposito["camellon"]
    dc_dobles    = datos_deposito["dobles"]

    # Procesamiento de arcos sin servicio
    da_camellon  = [(ar[1], ar[0], ar[2]) for ar in dc_camellon] + dc_camellon
    da_dobles    = [(ar[1], ar[0], ar[2]) for ar in dc_dobles]   + dc_dobles
    da_sencillos = dc_sencillos
    da_completos = da_sencillos + da_camellon + da_dobles

    deposito = datos_deposito["depot"]

    # ----------------------------------------------------------------
    # 3) Lista total de arcos y conjunto de nodos
    # ----------------------------------------------------------------
    arcos_totales = a_completos + da_completos
    nodos = set()
    for (n_o, n_d, _) in arcos_totales:
        nodos.add(n_o)
        nodos.add(n_d)
    nodos = list(nodos)

    # ----------------------------------------------------------------
    # 4) Crear el problema
    # ----------------------------------------------------------------
    problema = LpProblem("PPC_Mixto", LpMinimize)
    
    # ----------------------------------------------------------------
    # 5) Diccionario de costos
    # ----------------------------------------------------------------
    costo_arcos = {}
    for (i, j, cst) in arcos_totales:
        costo_arcos[(i, j)] = cst

    # ----------------------------------------------------------------
    # 6) Construcción de variables: x_{i,j,k}, f_{i,j,k}, delta_{i,k}
    # ----------------------------------------------------------------
    x = {}
    f = {}
    delta = {}
    
    # Variable alpha para la minimización del costo máximo (min-max)
    alpha = LpVariable("alpha", lowBound=0, cat="Continuous")
    
    # x_{i,j,k} y f_{i,j,k}
    for (i, j) in costo_arcos:
        for k in range(1, K+1):
            x[(i, j, k)] = LpVariable(f"x_{i}_{j}_k{k}", lowBound=0, cat="Integer")
            f[(i, j, k)] = LpVariable(f"f_{i}_{j}_k{k}", lowBound=0, cat="Continuous")
            
    # delta_{i,k}
    for i in nodos:
        for k in range(1, K+1):
            delta[(i, k)] = LpVariable(f"delta_{i}_{k}", lowBound=0, upBound=1, cat="Binary")

    # ----------------------------------------------------------------
    # 7) Función objetivo: Minimizar alpha
    # ----------------------------------------------------------------
    problema += alpha
    
    # ----------------------------------------------------------------
    # 8) Restricciones de cobertura (>=1)
    # ----------------------------------------------------------------
    # a) Arcos sencillos
    for (i, j, _) in a_sencillos:
        problema += lpSum(x[(i, j, k)] for k in range(1, K+1)) >= 1

    # b) Arcos dobles
    for (i, j, _) in a_dobles:
        problema += (
            lpSum(x[(i, j, k)] for k in range(1, K+1))
            + lpSum(x[(j, i, k)] for k in range(1, K+1))
            >= 1
        )
     
    # c) Arcos camellón
    for (i, j, _) in a_camellon:
        problema += lpSum(x[(i, j, k)] for k in range(1, K+1)) >= 1
        problema += lpSum(x[(j, i, k)] for k in range(1, K+1)) >= 1
        
    # ----------------------------------------------------------------
    # 9) Restricciones de salida desde el depósito
    # ----------------------------------------------------------------
    for k in range(1, K+1):
        out_0 = lpSum(
            x.get((deposito, j, k), 0)
            for j in nodos
            if (deposito, j, k) in x
        )
        problema += (out_0 >= 1)
           
    # ----------------------------------------------------------------
    # 10) Restricciones de flujo (in_x = out_x) por vehículo
    # ----------------------------------------------------------------
    for k in range(1, K+1):
        for n_i in nodos:
            in_x  = lpSum(x.get((u, n_i, k), 0) for u in nodos if (u, n_i, k) in x)
            out_x = lpSum(x.get((n_i, v, k), 0) for v in nodos if (n_i, v, k) in x)
            problema += (in_x == out_x)

    # ----------------------------------------------------------------
    # 11) Evitar subtours: x_{i,j,k} <= M * delta_{i,k}
    # ----------------------------------------------------------------
    M = len(nodos)
    for i in nodos:
        for j in nodos:
            for k in range(1, K+1):
                if (i, j, k) in x:
                    problema += x[(i, j, k)] <= M * delta[(i, k)]

    # ----------------------------------------------------------------
    # 12) Flujo auxiliar
    # ----------------------------------------------------------------
    for k in range(1, K+1):
        # Flujo en el depósito
        in_f_dep = lpSum(f.get((i, deposito, k), 0) for i in nodos if (i, deposito, k) in f)
        out_f_dep= lpSum(f.get((deposito, j, k), 0) for j in nodos if (deposito, j, k) in f)
        sum_deltas = lpSum(delta[(i, k)] for i in nodos if i != deposito)
        problema += (out_f_dep - in_f_dep == sum_deltas)
        
        # Flujo en nodos i != depósito
        for i in nodos:
            if i == deposito:
                continue
            in_fi = lpSum(f.get((m, i, k), 0) for m in nodos if (m, i, k) in f)
            out_fi= lpSum(f.get((i, n, k), 0) for n in nodos if (i, n, k) in f)
            problema += (out_fi - in_fi == -delta[(i, k)])
            
        # f_{i,j,k} <= M * x_{i,j,k}
        for (i, j) in costo_arcos.keys():
            problema += f[(i, j, k)] <= M * x[(i, j, k)]
        
    # ----------------------------------------------------------------
    # 13) Restringir el costo de cada vehículo <= alpha
    # ----------------------------------------------------------------
    for k in range(1, K+1):
        problema += lpSum(
            x[(i, j, k)] * costo_arcos[(i, j)] 
            for (i, j) in costo_arcos
        ) <= alpha
        
        # Agregamos aquí una restricción adicional (p.ej. cota superior)
        problema += lpSum(
            x[(i, j, k)] * costo_arcos[(i, j)]
            for (i, j) in costo_arcos
        ) <= 12010

    # ----------------------------------------------------------------
    # 14) Resolver
    # ----------------------------------------------------------------
    solver = PULP_CBC_CMD(
        msg=True,
        timeLimit=600,
        options=["ratioGap=0.5"],   
        keepFiles=True,
        logPath="miCBC.log"
    )
    estado = problema.solve(solver)

    # ----------------------------------------------------------------
    # 15) Recolectar rutas por vehículo
    # ----------------------------------------------------------------
    if estado == LpStatusOptimal:
        rutas_por_vehiculo = {k: [] for k in range(1, K+1)}
        
        for (i, j, k), var in x.items():
            valor = var.varValue
            if valor > 0.5:   
                rutas_por_vehiculo[k].extend([(i, j)] * int(valor))
        return rutas_por_vehiculo
    else:
        print("No hay solución factible.")
        return {}

# -------------------------------------------------------------------------------------
# Generación de grafo a partir de los arcos
# -------------------------------------------------------------------------------------
def grafo(arcos):
    g = {}
    for (a, b) in arcos:
        if a not in g:
            g[a] = []
        g[a].append(b)
    return g

# -------------------------------------------------------------------------------------
# Búsqueda en profundidad (DFS)
# -------------------------------------------------------------------------------------
def dfs(graph, node, eulerian_path):
    while graph.get(node, []):
        neighbor = graph[node].pop()
        dfs(graph, neighbor, eulerian_path)
    eulerian_path.append(node)

# -------------------------------------------------------------------------------------
# Construcción de camino euleriano
# -------------------------------------------------------------------------------------
def euler(grafo, deposito=None):
    if not grafo:
        print("El grafo está vacío, no hay camino Euleriano.")
        return []
    if deposito is None or deposito not in grafo:
        deposito = next(iter(grafo))
    eulerian_path = []
    dfs(grafo, deposito, eulerian_path)
    return eulerian_path[::-1]

# -------------------------------------------------------------------------------------
# Ejecución de prueba
# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    k = 4
    rutas_por_vehiculo = ppl(datos_region, datos_deposito, k)

    # Función auxiliar para construir diccionario de costos
    def construir_costos(datos_region, datos_deposito):
        region_sencillos = [
            arco for data in datos_region.values()
            for arco in data["sencillos"]
        ]
        region_camellon  = [
            arco for data in datos_region.values()
            for arco in data["camellon"]
        ]
        region_dobles    = [
            arco for data in datos_region.values()
            for arco in data["dobles"]
        ]

        a_camellon  = [(ar[1], ar[0], ar[2]) for ar in region_camellon] + region_camellon
        a_dobles    = [(ar[1], ar[0], ar[2]) for ar in region_dobles]   + region_dobles
        a_sencillos = region_sencillos
        a_completos = a_sencillos + a_camellon + a_dobles

        dc_sencillos = datos_deposito["sencillos"]
        dc_camellon  = datos_deposito["camellon"]
        dc_dobles    = datos_deposito["dobles"]

        da_camellon  = [(ar[1], ar[0], ar[2]) for ar in dc_camellon] + dc_camellon
        da_dobles    = [(ar[1], ar[0], ar[2]) for ar in dc_dobles]   + dc_dobles
        da_sencillos = dc_sencillos
        da_completos = da_sencillos + da_camellon + da_dobles

        arcos_totales = a_completos + da_completos

        costo_arcos = {}
        for (i, j, cost) in arcos_totales:
            costo_arcos[(i, j)] = cost
        return costo_arcos

    # Construir costos
    costo_arcos = construir_costos(datos_region, datos_deposito)

    # Mostrar ruta Euleriana y costos
    for vehiculo, lista_arcos in rutas_por_vehiculo.items():
        g = grafo(lista_arcos)
        euler_path = euler(g, deposito="0")
        
        # Calcular el costo de la ruta
        costo_ruta = 0
        for (i, j) in lista_arcos:
            costo_ruta += costo_arcos[(i, j)]
        
        print(f"\nVehículo {vehiculo}:")
        print("Camino Euleriano:", euler_path)
        print(f"Costo total de la ruta: {costo_ruta}")
