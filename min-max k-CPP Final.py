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
# Datos del grafo
# --------------------------------------------------------------------------------

# --- Datos para la región norte ---
norte_sencillos = []
norte_camellon  = [('5', '6', 120), ('6', '8', 60)]
norte_dobles    = [
    ('1', '2', 50), ('2', '3', 160), ('2', '5', 80), 
    ('3', '4', 40), ('3', '6', 50), ('7', '8', 60), ('8', '10', 50), 
    ('9', '10', 90), ('10', '11', 30), ('11', '12', 60), 
    ('11', '19', 120), ('19', '20', 90), ('5', '20', 170), 
    ('5', '13', 300), ('20', '21', 60), ('13', '14', 50), 
    ('13', '15', 30), ('20', '22', 100), ('15', '22', 190), 
    ('15', '18', 70), ('17', '18', 50), ('17', '16', 40), 
    ('18', '23', 110), ('17', '23', 160), ('19', '25', 340), 
    ('24', '25', 30), ('25', '26', 190), ('26', '27', 130), 
    ('26', '28', 60), ('22', '28', 390), ('23', '37', 560), 
    ('28', '29', 100), ('29', '30', 70), ('30', '31', 50), 
    ('30', '32', 70), ('32', '34', 50), ('32', '33', 60), 
    ('29', '35', 30), ('35', '36', 90), ('35', '37', 130),
    ('37', '38', 90), ('37', '40', 280), ('38', '40', 350), 
    ('38', '39', 240), ('40', '41', 200)
]

# --- Datos para la región centro_norte ---
centro_norte_sencillos = []
centro_norte_camellon  = [('48', '49', 170)]
centro_norte_dobles    = [
    ('42', '41', 180), ('41', '43', 170), ('43', '44', 40),
    ('43', '47', 320), ('44', '45', 170), ('45', '48', 330),
    ('49', '50', 70), ('69', '51', 610), ('46', '51', 250),
    ('45', '52', 190), ('52', '53', 180), ('53', '54', 90),
    ('53', '55', 40), ('55', '56', 80), ('55', '57', 30),
    ('57', '58', 100), ('57', '59', 180), ('59', '60', 40),
    ('60', '52', 270), ('60', '61', 90), ('61', '62', 60),
    ('61', '65', 80), ('62', '64', 80), ('62', '63', 40),
    ('59', '67', 30), ('59', '66', 30), ('66', '69', 50),
    ('67', '68', 20), ('68', '69', 70), ('67', '71', 160),
    ('68', '70', 110), ('66', '46', 730)
]

# --- Datos para la región centro_sur ---
centro_sur_sencillos = []
centro_sur_camellon  = []
centro_sur_dobles    = [
    ('72', '73', 340), ('44', '75', 430), ('75', '74', 70),
    ('75', '76', 50), ('76', '77', 70), ('77', '78', 60), 
    ('77', '79', 110), ('76', '80', 150), ('80', '83', 170), 
    ('80', '81', 70), ('81', '82', 260), ('81', '84', 240),
    ('84', '85', 130), ('84', '86', 110), ('86', '87', 200),
    ('86', '88', 100), ('88', '91', 240), ('88', '89', 40),
    ('89', '92', 230), ('91', '92', 40), ('91', '94', 220),
    ('92', '93', 130), ('89', '90', 120), ('90', '90', 540),
    ('88', '46', 810), ('89', '51', 840)
]

# --- Datos para la región sur ---
sur_sencillos = []
sur_camellon  = []
sur_dobles    = [
    ('95', '96', 100), ('96', '97', 50), ('96', '98', 200), 
    ('98', '99', 130), ('99', '94', 200), ('94', '100', 60), 
    ('100', '93', 80), ('100', '101', 130), ('99', '102', 160), 
    ('101', '102', 150), ('101', '106', 30), ('102', '103', 120), 
    ('103', '104', 160), ('105', '106', 340), ('106', '107', 110), 
    ('107', '108', 110), ('107', '110', 410), ('110', '109', 90), 
    ('110', '111', 190), ('99', '112', 1860), ('103', '112', 1400)
]

# --- Datos para el campamento ---
campamento_depot = "0"
depot_sencillos = [
    ('0', '112', 3870), ('112', '98', 2020), ('98', '82', 900), 
    ('82', '72', 150), ('42', '0', 1200), ('72', '42', 170),
    ('0', '24', 1600),  ('42', '24', 710)
]
depot_camellon  = []
depot_dobles    = [('0', '98', 1770)]

# --------------------------------------------------------------------------------
# Diccionarios por región
# --------------------------------------------------------------------------------
datos_region = {
    "norte": {
        "sencillos": norte_sencillos,
        "camellon":  norte_camellon,
        "dobles":    norte_dobles,
    },
    "centro_norte": {
        "sencillos": centro_norte_sencillos,
        "camellon":  centro_norte_camellon,
        "dobles":    centro_norte_dobles,
    },
    "centro_sur": {
        "sencillos": centro_sur_sencillos,
        "camellon":  centro_sur_camellon,
        "dobles":    centro_sur_dobles,
    },
    "sur": {
        "sencillos": sur_sencillos,
        "camellon":  sur_camellon,
        "dobles":    sur_dobles,
    },
}

datos_deposito = {
    "sencillos": depot_sencillos,
    "camellon":  depot_camellon,
    "dobles":    depot_dobles,
    "depot":     campamento_depot,
}

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
