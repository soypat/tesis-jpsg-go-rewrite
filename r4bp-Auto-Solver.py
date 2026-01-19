#######################
#####  LIBRERIAS  ####
#######################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import pandas as pd
import multiprocessing as mp
from itertools import product
import os
plt.rcParams.update({'font.size': 14})

#######################
#####  CONSTANTES  ####
#######################
days = 24 * 3600                   # segundos/día
G = 6.6742e-20                     # km^3/kg/s^2
rmoon = 1737                       # km
rearth = 6378                      # km
r12 = 384400                       # km distancia Tierra-Luna

m1 = 5.974e24                      # kg (Tierra)
m2 = 7.348e22                      # kg (Luna)
M = m1 + m2
pi_1 = m1 / M
pi_2 = m2 / M

mu1 = 398600.0                     # km^3/s^2 (Tierra)
mu2 = 4903.02                      # km^3/s^2 (Luna)
mu = mu1 + mu2

# Parámetros del Sol
mS = 1.989e30                      # kg (Sol)
muS = G * mS                       # km^3/s^2 (Sol)
R_B2S = 149597870.7                # km (baricéntrico T-L-Sol)
nS = np.sqrt(muS / R_B2S**3)       # rad/s

# Velocidad angular del sistema Tierra-Luna
W = np.sqrt(mu / r12**3)           # rad/s

# Posiciones del baricentro T-L en rotante
x1 = -pi_2 * r12                   # Tierra
x2 =  pi_1 * r12                   # Luna

# Umbrales
C1 = -1.676                        # Jacobi para fase de frenado
L1 = 321710                        # km distancia L1 al centro terrestre


################################
#  Parámetros de empuje (globales)
################################
n = 4                              # Numero de motores
F = 0.00000045                     # kN (Empuje de cada motor)
T_val = F * n                      # Empuje del sistema
tol = 1e-12                        # Tolerancia es de un coso que me olvide


################################
#  Funcion para el potencial de Jacobi
################################
def jacobi_potential(x, y):
    r1 = np.hypot(x + pi_2 * r12, y)
    r2 = np.hypot(x - pi_1 * r12, y)
    return -0.5 * W**2 * (x**2 + y**2) - mu1 / r1 - mu2 / r2

################################
# PseudoPotencial de Jacobi en 4 cuerpos
################################
def jacobi_potential_4b(x, y,phiS):

    # Distancias a Tierra y Luna
    r1 = np.hypot(x + pi_2*r12, y)
    r2 = np.hypot(x - pi_1*r12, y)
    U_em = -0.5*W**2*(x**2 + y**2) - mu1/r1 - mu2/r2

    # Posición del Sol en el marco TL rotante
    xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
    rS = np.hypot(x - xS, y - yS)

    # Potencial solar 
    U_sun_direct   = -muS/rS + muS/R_B2S
    U_sun_indirect = +muS*(x*xS + y*yS)/R_B2S**3        # ax_ind = +uS xS /R**3 --> U_indirect = −uS x xS /R**3

    return U_em + U_sun_direct + U_sun_indirect

################################
#  Eventos
################################
def jacobiC(t, state,phiS):
    x_val, y_val, vx, vy, _ = state
    v_val = np.hypot(vx, vy)
    r1_val = np.hypot(x_val + pi_2 * r12, y_val)
    r2_val = np.hypot(x_val - pi_1 * r12, y_val)
   
    #Posicion del Sol
    xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
    rS = np.hypot(x_val-xS , y_val-yS)

    U_em = 0.5 * v_val**2 - 0.5 * W**2 * (x_val**2 + y_val**2) - mu1 / r1_val - mu2 / r2_val 
    U_S_direct = -muS/rS + muS/R_B2S
    U_S_indirect = muS*(x_val*xS + y_val*yS)/R_B2S**3

    # Umbral fijo - Empleado durante debugging
    thr = -1.625#-1.63907788
    return U_em + U_S_direct + U_S_indirect - thr
jacobiC.terminal = True
jacobiC.direction = 0

Y_WINDOW_L1 = 55000.0  # km

def lagranian1(t, state, phiS0_rad):
    """
    Dispara cuando la trayectoria cruza la línea x = x_L1 garganta de L1,
    pero solo si |y| <= Y_WINDOW_L1. Fuera de esa franja en Y, la función
    se mantiene lejos de cero para evitar raíces espurias.
    """
    x_val, y_val, *_ = state

    x_L1 = x1 + L1
    s = x_val - x_L1  # distancia horizontal a la línea de L1

    if abs(y_val) <= Y_WINDOW_L1:
        # Dentro de la franja vertical: raíz cuando x cruza x_L1
        return s
    eps = 1e-6  # tolerancia para evitar problemas
    margin = (abs(y_val) - Y_WINDOW_L1) + eps
    return (1.0 if s >= 0.0 else -1.0) * margin

lagranian1.terminal = True
lagranian1.direction = 0

def jacobiC1(t, state,phiS0_rad):
    x_val, y_val, vx, vy, _ = state
    v_val = np.hypot(vx, vy)
    r1_val = np.hypot(x_val + pi_2 * r12, y_val)
    r2_val = np.hypot(x_val - pi_1 * r12, y_val)
    phiS = phiS0_rad 
    #Posicion del Sol
    xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
    rS = np.hypot(x_val-xS , y_val-yS)

    U_em = 0.5 * v_val**2 - 0.5 * W**2 * (x_val**2 + y_val**2) - mu1 / r1_val - mu2 / r2_val 
    U_S_direct = -muS/rS + muS/R_B2S
    U_S_indirect = muS*(x_val*xS + y_val*yS)/R_B2S**3

    # Umbral fijo C1
    return U_em + U_S_direct + U_S_indirect - C1
jacobiC1.terminal = True
jacobiC1.direction = 0

def circular_event(t, state):
    x_val, y_val, vx, vy, _ = state
    r2_vec = np.array([x_val - pi_1 * r12, y_val])
    v_vec = np.array([vx, vy])
    return np.dot(r2_vec, v_vec)
circular_event.terminal = True
circular_event.direction = 0

def collision_event(t, state,phiS0_rad):
    x_val, y_val, *_ = state
    # Distancia al centro lunar
    d_moon = np.hypot(x_val - x2, y_val)
    return d_moon - rmoon
collision_event.terminal = True
collision_event.direction = 0


def capture_event(t, state,phiS0_rad):
    x_val, y_val, vx, vy, _ = state
    r_rel = np.hypot(x_val - x2, y_val)
    speed = np.hypot(vx, vy)
    E = 0.5 * speed**2 - mu2 / r_rel
    return E
capture_event.terminal = True
capture_event.direction = -1


#############################
#  Funciones de propagacion
#############################
def rates(t, state, phiS0_rad):
    """
    Fase con empuje usando T_val 
    """
    x, y, vx, vy, m = state

    # Distancias a Tierra y Luna
    r1_val = np.hypot(x + pi_2 * r12, y)
    r2_val = np.hypot(x - pi_1 * r12, y)
    v_val = np.hypot(vx, vy)


    # Pos del SOl en marco T-L
    # En el marco rotante Tierra–Luna, el Sol gira con velocidad angular (nS - W)
    phiS = phiS0_rad + (nS - W) * t
    xS = R_B2S * np.cos(phiS)
    yS = R_B2S * np.sin(phiS)
    # Distancia nave-Sol
    dxS = x - xS
    dyS = y - yS
    rS_val = np.hypot(dxS, dyS)

    # Aceleraciones: Coriolis, centrifuga, Tierra, Luna, Sol, y empuje
    ax = (  2 * W * vy
            + W**2 * x
            - mu1 * (x - x1) / (r1_val**3)
            - mu2 * (x - x2) / (r2_val**3)
            # Termino directo de la gravedad solar:
            - muS * dxS / (rS_val**3)
            # Termino indirecto aceleracion del baricentro por el Sol:
            - muS * xS / (R_B2S**3)
            + (T_val / m) * (vx / v_val) )

    ay = ( -2 * W * vx
            + W**2 * y
            - (mu1 / (r1_val**3) + mu2 / (r2_val**3)) * y
            # Termino directo de la gravedad solar:
            - muS * dyS / (rS_val**3)
            # Termino indirecto aceleracion del baricentro por el Sol:
            - muS * yS / (R_B2S**3)
            + (T_val / m) * (vy / v_val) )

    # Caudal másico mdot
    g0 = 9.807     # m/s²
    Isp = 1650     # s
    mdot = -T_val * 1000.0 / (g0 * Isp)

    return [vx, vy, ax, ay, mdot]


def rates0(t, state, phiS0_rad):
    """
    Fase sin empuje (coasting)
    """
    x, y, vx, vy, m = state

    # Distancias a Tierra y Luna
    r1_val = np.hypot(x + pi_2 * r12, y)
    r2_val = np.hypot(x - pi_1 * r12, y)
    v_val = np.hypot(vx, vy)

    # Pos del SOl en marco T-L
    # En el marco rotante Tierra–Luna, el Sol gira con velocidad angular (nS - W)
    phiS = phiS0_rad + (nS - W) * t
    xS = R_B2S * np.cos(phiS)
    yS = R_B2S * np.sin(phiS)
    # Distancia nave-Sol
    dxS = x - xS
    dyS = y - yS
    rS_val = np.hypot(dxS, dyS)

    # Aceleraciones: Coriolis, centrifuga, Tierra, Luna, Sol, y empuje
    ax = (  2 * W * vy
            + W**2 * x
            - mu1 * (x - x1) / (r1_val**3)
            - mu2 * (x - x2) / (r2_val**3)
            # Termino directo de la gravedad solar:
            - muS * dxS / (rS_val**3)
            # Termino indirecto aceleracion del baricentro por el Sol:
            - muS * xS / (R_B2S**3)
            )

    ay = ( -2 * W * vx
            + W**2 * y
            - (mu1 / (r1_val**3) + mu2 / (r2_val**3)) * y
            # Termino directo de la gravedad solar:
            - muS * dyS / (rS_val**3)
            # Termino indirecto aceleracion del baricentro por el Sol:
            - muS * yS / (R_B2S**3)
             )

    # Caudal masico mdot
    mdot = 0

    return [vx, vy, ax, ay, mdot]


def rates_B(t, state, phiS0_rad):
    """
    Fase con empuje (positiva) usando T_val constante 
    """
    x, y, vx, vy, m = state

    # Distancias a Tierra y Luna
    r1_val = np.hypot(x + pi_2 * r12, y)
    r2_val = np.hypot(x - pi_1 * r12, y)
    v_val = np.hypot(vx, vy)

    # Pos del SOl en marco T-L
    # En el marco rotante Tierra–Luna, el Sol gira con velocidad angular (nS - W)
    phiS = phiS0_rad + (nS - W) * t
    xS = R_B2S * np.cos(phiS)
    yS = R_B2S * np.sin(phiS)
    # Distancia nave-Sol
    dxS = x - xS
    dyS = y - yS
    rS_val = np.hypot(dxS, dyS)
    T_neg = -1*T_val

    # Aceleraciones: Coriolis, centrifuga, Tierra, Luna, Sol, y empuje
    ax = (  2 * W * vy
            + W**2 * x
            - mu1 * (x - x1) / (r1_val**3)
            - mu2 * (x - x2) / (r2_val**3)
            # Termino directo de la gravedad solar:
            - muS * dxS / (rS_val**3)
            # Termino indirecto aceleracion del baricentro por el Sol:
            - muS * xS / (R_B2S**3)
            + (T_neg / m) * (vx / v_val) )

    ay = ( -2 * W * vx
            + W**2 * y
            - (mu1 / (r1_val**3) + mu2 / (r2_val**3)) * y
            # Termino directo de la gravedad solar:
            - muS * dyS / (rS_val**3)
            # Termino indirecto aceleracion del baricentro por el Sol:
            - muS * yS / (R_B2S**3)
            + (T_neg / m) * (vy / v_val) )

    # Caudal masico mdot
    g0 = 9.807     # m/s²
    Isp = 1650     # s
    mdot = -T_val * 1000.0 / (g0 * Isp)

    return [vx, vy, ax, ay, mdot]

# -----------------------------
#  Función para dibujar un círculo
# -----------------------------
def circle(xc, yc, radius, num_points=361):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    return x, y

def plot_transfer(sol1,sol2,markers, title="Transferencia Tierra-Luna ", xlim=(-400000, 450000), ylim=(-325000, 325000)):

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Trayectorias    
    ax.plot(sol1.y[0], sol1.y[1], '-', color='g', label='Fase 1 (Empuje)')
    ax.plot(sol2.y[0], sol2.y[1], '-', color='r', label='Fase 2 (Coasting)')

    # 2) Tierra y Luna
    earth_x, earth_y = circle(x1, 0, rearth)
    moon_x,  moon_y  = circle(x2, 0, rmoon)
    ax.fill(earth_x, earth_y, 'b', alpha=0.8, label='Tierra')
    ax.fill(moon_x,  moon_y,  'gray', alpha=0.8, label='Luna')

    # 3) Marcadores del Sol
    for m in markers:
        ax.scatter([m['x']], [m['y']],
                   marker=m.get('marker','*'),
                   s=m.get('s',150),
                   color=m.get('color','gold'),
                   edgecolors='k',
                   label=m['label'])
    ax.axvline(x=L1,
               linestyle='--',
               linewidth=1.2,
               label='L₁')

    # 4) Ajustes finales
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.show()

    return fig, ax

def save_result(result):
    df = pd.DataFrame([result])
    filename = 'resultados_trayectorias4bp.csv'
    if not os.path.exists(filename):
        df.to_csv(filename, index= False)
    else:
        df.to_csv(filename,mode='a',header=False,index=False)

#################################
#  Funcion principal de simulacion
################################

def trayectoria(params):
    phi,phiS0,d0,jacobiTHR = params
    start_time = time.time()

    # 1) Condiciones iniciales
    d0 = 37000                                   # km
    r0 = rearth + d0                             # km 
    v0 = np.sqrt(mu1 / r0) - W * r0              # km/s
    gamma = 0                                    # grados

    # Radianes
    phi_rad    = np.deg2rad(phi)
    gamma_rad  = np.deg2rad(gamma)
    phiS0_rad  = np.deg2rad(phiS0)
    r_mark = 300000
    xSmark0, ySmark0 = r_mark*np.cos(phiS0_rad), r_mark*np.sin(phiS0_rad)

    # Posicion inicial rotante
    x0 = r0 * np.cos(phi_rad) + x1
    y0 = r0 * np.sin(phi_rad)

    # Descomposicion de v0
    vx0 = v0 * (np.sin(gamma_rad) * np.cos(phi_rad)
                - np.cos(gamma_rad) * np.sin(phi_rad))
    vy0 = v0 * (np.sin(gamma_rad) * np.sin(phi_rad)
                + np.cos(gamma_rad) * np.cos(phi_rad)) 

    m0_val = 12.0  # masa inicial (kg)
    f0 = [x0, y0, vx0, vy0, m0_val]

    # Duraciones de fases
    t0 = 0
    tf = days * 360*4  # 4 años para la fase 1
    def jacobiC_local(t, state,phiS):
        x_val, y_val, vx, vy, _ = state
        v_val = np.hypot(vx, vy)
        r1_val = np.hypot(x_val + pi_2 * r12, y_val)
        r2_val = np.hypot(x_val - pi_1 * r12, y_val)
    
        #Posicion del Sol
        xS, yS = R_B2S*np.cos(phiS), R_B2S*np.sin(phiS)
        rS = np.hypot(x_val-xS , y_val-yS)

        U_em = 0.5 * v_val**2 - 0.5 * W**2 * (x_val**2 + y_val**2) - mu1 / r1_val - mu2 / r2_val 
        U_S_direct = -muS/rS + muS/R_B2S
        U_S_indirect = muS*(x_val*xS + y_val*yS)/R_B2S**3

        # Umbral fijo
        thr = -1.639#-1.63907788
        return U_em + U_S_direct + U_S_indirect - jacobiTHR#jacobiTHR
    jacobiC_local.terminal = True
    jacobiC_local.direction = 0

    exito = True
    capture_success = False

    # Fase 1: Empuje ()
    sol1 = solve_ivp(
        rates,
        [t0, tf],
        f0,
        args=(phiS0_rad,),
        method='RK45',
        events=[jacobiC_local],        # Lista de eventos
        rtol=1e-9,
        atol=tol,
        max_step=450
    )
    print("Fase 1 completada, t_eventos:", sol1.t_events, "shape:", sol1.y.shape)
    if sol1.t_events[0].size > 0:
        print(f"[Fase 1] Evento jacobiC_local disparado en t = {sol1.t_events[0]} con Jacobi")
        print(f"[Fase 1] Estado en evento: {sol1.y_events[0]}")
    else:
        print("[Fase 1] No se disparó el evento jacobiC_local")
    f1_final = sol1.y[:, -1]
    t_fase01 = sol1.t_events[0][0]
    phiS0_rad_2 = phiS0_rad - (nS - W)*t_fase01
    xSmark1, ySmark1 = r_mark*np.cos(phiS0_rad_2), r_mark*np.sin(phiS0_rad_2)

    # Fase 2: Coasting (motores apagados)
    t_phase2 = [sol1.t[-1], sol1.t[-1] + days * 650]
    sol2 = solve_ivp(rates0, 
            t_phase2,
            f1_final, 
            args=(phiS0_rad_2,),
            method='RK45',
            events=lagranian1,
            rtol=1e-9, 
            atol=tol, 
            max_step=100
            )
    print("Fase 2 completada, tiempo de evento:", sol2.t_events, sol2.y.shape)
    if sol2.t_events[0].size == 0:
        print('No se logró llegar a L1')
        exito = False
        f2_final = sol2.y[:, -1]
        tiempo_total = sol2.t[-1]
        masa_final = sol2.y[4, -1]
        detalles = {
            'phi': phi,
            'jacobi_thr': jacobiTHR,
            'phiS0':phiS0,
            'd0': d0,
            'tiempo_total': tiempo_total,
            'masa_final': masa_final,
            'exito': exito,
            'SOI': False,
            'time_exec': time.time() - start_time
        }
        return detalles
    else:
        print('Se alcanzo L1 CORRECTAMENTE')
        f2_final = sol2.y[:, -1]

    t_fase02 = sol2.t[-1]
    phiS0_rad_3 = phiS0_rad - (nS - W)*t_fase02
    xSmark2, ySmark2 = r_mark*np.cos(phiS0_rad_3), r_mark*np.sin(phiS0_rad_3)

    # Fase 3: Frenado para inserción lunar
    t_phase3 = [sol2.t[-1], sol2.t[-1] + days * 180]
    sol3 = solve_ivp(rates_B,
            t_phase3,
            f2_final,
            args=(phiS0_rad_3,),
            method='RK45',
            events=[jacobiC1,collision_event, capture_event], 
            rtol=1e-9, 
            atol=tol, 
            max_step=100
            )
    
    d_phase2 = np.sqrt((sol2.y[0] - x2)**2 + (sol2.y[1])**2)
    d_phase3 = np.sqrt((sol3.y[0] - x2)**2 + (sol3.y[1])**2)
    min_distance = min(np.min(d_phase2), np.min(d_phase3))
    SOI_flag = (min_distance <= 55000)
    

    if len(sol3.t_events[1]) > 0:
        if len(sol3.t_events[1]) > 0 : print('Choco la LUNA phi -',phi,)
        exito = False
        tiempo_total = sol2.t[-1]
        masa_final = sol2.y[4,-1]
    else:
        t_phase3 = sol3.t
        x_phase3 = sol3.y[0]
        y_phase3 = sol3.y[1]
        vx_phase3 = sol3.y[2]
        vy_phase3 = sol3.y[3]

        # Calcular la distancia relativa a la Luna
        r_rel_phase3 = np.sqrt((x_phase3 - x2)**2 + (y_phase3)**2)

        # Calcular la energía de captura en cada instante
        E_capture = 0.5 * (vx_phase3**2 + vy_phase3**2) - mu2 / r_rel_phase3

        # Verificar si en todo el intervalo se cumple E < 0
        if np.all(E_capture < 0):
            print("La condición de captura (E < 0) se cumple - phi", phi)
            capture_success = True
            exito = True
            tiempo_total = sol3.t[-1]
            masa_final = sol3.y[4, -1]
        else:
            print("La condición de captura NO - ", phi)
            capture_success = False
            exito = False
            tiempo_total = sol3.t[-1]
            masa_final = sol3.y[4, -1]
        

    # print("Fase 3 completada, tiempo de evento:", sol3.t_events, sol3.y.shape)
    f3_final = sol3.y[:, -1]

    t_fase03 = sol3.t[-1]
    phiS0_rad_4 = phiS0_rad - (nS - W)*t_fase03
    xSmark3, ySmark3 = r_mark*np.cos(phiS0_rad_4), r_mark*np.sin(phiS0_rad_4)



    end_time = time.time()
    detalles = {
            'phi': phi,
            'jacobi_thr': jacobiTHR,
            'phiS0':phiS0,
            'd0': d0,
            'tiempo_total': tiempo_total/days,
            'masa_final': masa_final,
            'exito': exito,
            'SOI': SOI_flag,
            'time_exec': time.time() - start_time
    }
    return detalles
   


# -----------------------------
# Iterar 
# -----------------------------
def iterar_parametros():
    # Params def trayectoria(params):
    # phi,phiS0,d0,thr = params
    phis = np.linspace(295,296,50)
    phiS0 = [0]#np.linspace(0,360,5)
    jacobi_threshold = [-1.639]#[-1.63907788]
    d0_vals = [37000]
    
    # Crear todas las combinaciones 
    parametros = list(product(phis, phiS0, d0_vals, jacobi_threshold))
    
    resultados=[]
    async_results = []

    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        for param in parametros:
            async_result = pool.apply_async(trayectoria, args=(param,), callback=save_result)
            async_results.append(async_result)
            #resultados = pool.map(trayectoria, parametros)
        for async_result in async_results:
            try:
                res = async_result.get()
                resultados.append(res)
            except Exception as e:
                print('Error iter ', e)
    resumen = []

    for res in resultados:
        resumen.append({
            'phi': res['phi'],
            'jacobi_thr': res['jacobi_thr'],
            'phiS0':res['phiS0'],
            'd0': res['d0'],
            'tiempo_total': res['tiempo_total'],
            'masa_final': res['masa_final'],
            'exito': res['exito'],
            'SOI': res['SOI'],
            'time_exec': res['time_exec']
        })

    df = pd.DataFrame(resultados)
    df.to_csv('resultados_trayectorias4bp.csv', index=False)
    print(df)
    print("Resultados guardados en 'resultados_trayectorias4bp.csv'")
    return df

########################
#  Ejecución principal
########################
if __name__ == '__main__':
    df_resultados = iterar_parametros()
    print(df_resultados)
