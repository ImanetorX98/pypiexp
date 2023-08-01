import random
from sympy import *
import numpy as np
from rosignoli_lib import *

# Dichiarazione delle costanti utili alla simulazione come variabli globali

global convfac
global alpha
global PI
global e2
global me
global me2
global mm
global mm2
global ZRM

convfac = 3.8937e8 # Conversione tra GeV**2 e pb
alpha = 1 / 137
PI = 3.14159265
e2 = 4*PI*alpha
me = .511e-3
me2 = me**2
mm = .106
mm2 = mm**2
ZRM = 1e-35

def minkowski_dot(v1, v2):
    if len(v1) != 4 or len(v2) != 4:
        raise ValueError("I vettori devono avere 4 componenti (tempo + 3 spaziali)")
    
    # Metrica di Minkowski
    eta = [1, -1, -1, -1]
    
    # Calcolo del prodotto scalare
    dot_product = sum(eta[i] * v1[i] * v2[i] for i in range(4))
    return dot_product

def generate_randoms(estremi = [-1,1]):
    x1 = random.random()
    x2 = random.random()
    cos = random.uniform(-1, 1)
    phi = random.uniform(0, 2*np.pi)
    sez = random.uniform(estremi[0], estremi[1])
    return phi,cos,sez,x1,x2

def matrix_element(p1,p2,q1,q2):
    s = minkowski_dot(p1+p2, p1+p2)
    t = minkowski_dot(p1-q1, p1-q1)
    u = minkowski_dot(p1-q2, p1-q2)
    # For spinless electrons
    # emtrx = t*t+u*u-2.d0*u*t
    # For 1/2 hbar spin electrons
    emtrx = (t ** 2 + u ** 2) * s ** (-2)
    return emtrx

def D(x, s):
    beta = 2.0 * alpha / np.pi * (np.log(s / me2) - 1.0)
    betao2 = beta / 2.0
    if x == 1:
        return 2 * beta
    else:
        return betao2 * (1.0 - x)**(betao2 - 1.0) + beta * (1.0 + x)

def Dop(x, s):
    beta = 2.0 * alpha / np.pi * (np.log(s / me2) - 1.0)
    betao2 = beta / 2.0
    if x == 1:
        return betao2 + 1
    else:
        return 1 + beta * (1 + x) / (betao2 * (1 - x) ** (betao2 - 1))

def boost(p, q):
    """
    Esegue la trasformazione di boost relativistica sul quadri-vettore p dato la velocità di boost beta.
    """

    if len(p) != 4 or len(q) != 4:
        raise ValueError("Il quadri-vettore deve avere 4 componenti (tempo + 3 spaziali) e beta deve avere 4 componenti (tempo + 3 spaziali)")

    norm3q = np.linalg.norm(q[1:3]) 

    if q[0] > 0 and norm3q > 0:
        betaq = norm3q / q[0]
        if betaq > 1:
            gamma = 0
        else:
            gamma = 1 / np.sqrt(1-betaq**2)
    else:
        betaq = 0
        gamma = 1
        p_boosted = p
        return p_boosted

    AN = - q[1:3] / norm3q
    QJSLM = np.dot(p,AN)
    QJLS = QJSLM * AN
    QJT = p[1:3] - QJLS
    p_boosted[0] = gamma * (p[0] -  betaq * QJSLM)
    QJLM = gamma * (-betaq * p[0] + QJSLM)
    p_boosted[1:3] = QJLM * AN + QJT
    return p_boosted

def DPHI2(SHAT, CTHJ, PHIJ):
    global mm2
    QI2 = SHAT
    QJ2 = mm2
    QK2 = mm2
    # Calcola flusso e 4-momenti
    QJ = np.zeros(4)
    QK = np.zeros(4)

    STHJ = np.sqrt(1.0 - CTHJ ** 2)
    CPHIJ = np.cos(PHIJ)
    SPHIJ = np.sin(PHIJ)

    QJM2 = (QI2 + QK2 - QJ2) * (QI2 + QK2 - QJ2) / (4.0 * QI2) - QK2
    IKINEI = 0
    if QJM2 < 0.0:
        IKINEI += 1
        # print(' ')
        # print('MASSA INV FUORI RANGE: ', QJM2)
        QJM2 = 1.0e-13

    QJM = np.sqrt(QJM2)

    # MOMENTA ARE CALCULATED
    QJ = np.zeros(4)
    QJ[0] = np.sqrt(QJ2 + QJM2)
    QJ[1] = QJM * STHJ * CPHIJ
    QJ[2] = QJM * STHJ * SPHIJ
    QJ[3] = QJM * CTHJ

    QK = np.zeros(4)
    QK[0] = np.sqrt(QK2 + QJ[1]*QJ[1] + QJ[2]*QJ[2] + QJ[3]*QJ[3])
    QK[1] = -QJ[1]
    QK[2] = -QJ[2]
    QK[3] = -QJ[3]

    # FLUX IS CALCULATED
    FLUX = QJM / (QJ[0] + QK[0])

    return QJ, QK, FLUX


if __name__ == "__main__":
    try:
        ebeam = int(chiedi_numero("L'energia del fascio in GeV = "))
        nhitwmax = int(chiedi_numero_pos("Si inserisca il numero massimo di chiamate per eventi pesati = "))
        nhitmax = int(chiedi_numero_pos("Si inserisca il numero massimo di chiamate per eventi non pesati = "))
        perc = int(chiedi_numero_interval("Percentuale del valore massimo dell'integrale che utilizziamo per generare punti = ", 0, 100)) / 100
        crct = chiedi_conferma("Si implementa il campionamento di importanza? S o N : ")
        ncalls = int(1e9)
        s = 4 * ebeam ** 2
        rs = 2 * ebeam
        beta = 2 * alpha / PI * (np.log(s/me2) - 1)
        betao2 = beta / 2
        twoobeta = 2 / beta
        
        FUNVALMAX = -100
        FUNVALMIN = +100
        FAV = 0
        DFAV = 0
        FSUM = 0
        F2SUM= 0
        NHITW = 0
        encnt = 0
        
        nmax = ncalls
        icnt = 0
        funval = 0
        EMTRX = 1
        # Si parte con la simulazione wieghted
        # Chiamiamo la generazione di numeri random
        for i in range(nmax):
            icnt += 1
            funval = 0
            CMV = generate_randoms() # [0] è phi, [1] è cos
            cphi = np.cos(CMV[0])
            sphi = np.sin(CMV[0])
            cth = CMV[1]
            sth = np.sqrt(1-cth**2)
            if crct:
                x1 = 1 - (1 - CMV[3]) ** (twoobeta)
                x2 = 1 - (1 - CMV[4]) ** (twoobeta)
            else:
                x1 = CMV[3]
                x2 = CMV[4]
            #print(x1,x2)
            shat = x1*x2*s
            rshat = np.sqrt(shat)
            if shat < 4 * mm2:
                encnt += 1
                #print("Not enough energy to produce two rest muons")
                print("\rNot enough energy to produce two rest muons " + str(encnt) + " ", end='', flush=True)
                continue
            p1 = np.array([x1*ebeam,0,0,x1*ebeam])
            p2 = np.array([x2*ebeam,0,0,-x2*ebeam])
            p1s = np.array([rshat/2,0,0,rshat/2])
            p2s = np.array([rshat/2,0,0,-rshat/2])
            IKINE = 0
            q1s, q2s, F12 = DPHI2(shat, cth, CMV[0])
            qboost = np.array([(x1+x2)*ebeam,0,0,(x1-x2)*ebeam])
            q1 = boost(q1s,qboost)
            q2 = boost(q2s,qboost)
            JAC = 2*np.pi * 2 * F12
            emtrx = matrix_element(p1s,p2s,q1s,q2s)
            emtrx *= e2 ** 2
            FUNVAL = 1 * convfac / ((2*PI) ** 6 * 2 ** 2) * (2*PI) ** 4 / 2 / shat
            if crct:
                FUNVAL *= JAC * emtrx * Dop(x1,s) * Dop(x2,s)
            else:
                FUNVAL *= JAC * emtrx * D(x1,s) * D(x2,s)
            if FUNVAL > FUNVALMAX:
                FUNVALMAX = FUNVAL
            if FUNVAL < FUNVALMIN:
                FUNVALMIN = FUNVAL
            NHITW += 1
            if NHITW > nhitwmax:
                print("\nPiù chiamate delle chiamate massime")
                break
            FSUM += FUNVAL
            F2SUM += FUNVAL**2
            
        print(f'\nicnt = {icnt}')
        AN = icnt
        FAV = FSUM / AN
        F2AV = F2SUM / AN
        if AN > 1:
            DFAV = np.sqrt(F2AV - FAV**2)/np.sqrt(AN-1)
        else:
            DFAV = 0
        print(f'XSECT (wighted events) = {FAV} +- {DFAV} PB')
        print(f'FUNVALMAX = {FUNVALMAX}')
        #print(f'FUNVALMIN = {FUNVALMIN}')
        # Si comincia a generare eventi unweighted
        FUNVALMAX *= perc # 10
        FAV = 0
        DFAV = 0
        FSUM = 0
        F2SUM = 0
        
        nmax = ncalls
        icnt = 0
        encnt = 0
        FUNVAL = 0
        NHIT = 0
        NMIS = 0
        nbyas = 0
        emtrx = 1
        
        file_path_1 = "data_1_1.txt"
        file_path_2 = "data_1_2.txt"
            
        for i in range(nmax):
            icnt += 1
            funval = 0
            CMV = generate_randoms([0, FUNVALMAX]) # [0] è phi, [1] è cos
            cphi = np.cos(CMV[0])
            sphi = np.sin(CMV[0])
            cth = CMV[1]
            sth = np.sqrt(1-cth**2)
            rnumber = CMV[2]
            if crct:
                x1 = 1 - (1 - CMV[3]) ** (twoobeta)
                x2 = 1 - (1 - CMV[4]) ** (twoobeta)
            else:
                x1 = CMV[3]
                x2 = CMV[4]
            #print(x1,x2)
            shat = x1*x2*s
            rshat = np.sqrt(shat)
            if shat < 4 * mm2:
                encnt += 1
                #print("Not enough energy to produce two rest muons")
                print("\rNot enough energy to produce two rest muons " + str(encnt) + " ", end='', flush=True)
                continue
            p1 = np.array([x1*ebeam,0,0,x1*ebeam])
            p2 = np.array([x2*ebeam,0,0,-x2*ebeam])
            p1s = np.array([rshat/2,0,0,rshat/2])
            p2s = np.array([rshat/2,0,0,-rshat/2])
            IKINE = 0
            q1s, q2s, F12 = DPHI2(shat, cth, CMV[0])
            qboost = np.array([(x1+x2)*ebeam,0,0,(x1-x2)*ebeam])
            q1 = boost(q1s,qboost)
            q2 = boost(q2s,qboost)
            JAC = 2*np.pi * 2 * F12
            emtrx = matrix_element(p1s,p2s,q1s,q2s)
            emtrx *= e2 ** 2
            FUNVAL = 1 * convfac / ((2*PI) ** 6 * 2 ** 2) * (2*PI) ** 4 / 2 / shat
            if crct:
                FUNVAL *= JAC * emtrx * Dop(x1,s) * Dop(x2,s)
            else:
                FUNVAL *= JAC * emtrx * D(x1,s) * D(x2,s)
            if FUNVAL > FUNVALMAX: #or rnumber < FUNVALMIN:
                nbyas += 1

            if rnumber > FUNVAL:
                NMIS += 1
                continue
            NHIT += 1
            
            if NHIT > nhitmax:
                print("\nPiù chiamate delle chiamate massime")
                break

            # Numbers to write in a single line
            numbers = [NHIT, ebeam, x1, x2, p1, p2, q1, q2]

            # Convert numbers to strings and concatenate them with a space between each number
            numbers_str = ' '.join(str(num) for num in numbers)

            with open(file_path_1, 'a') as file_1:
                file_1.write("\n" + numbers_str)

            result_1 = q1[3] / np.sqrt(np.sum(q1[1:3]**2))
            result_2 = q2[3] / np.sqrt(np.sum(q2[1:3]**2))
            
            with open(file_path_2, 'a') as file_2:
                file_2.write(f'\n{result_1} {result_2}')
                
        eff = NHIT / icnt
        effm = NMIS / icnt
        effb = nbyas / NHIT
        hmint = eff * FUNVALMAX
        ehmint= FUNVALMAX / np.sqrt(icnt) * np.sqrt(eff * (1 - eff))
        print(f'\nResults from MC unweighted integration')
        print(f'n. of points = {icnt}    integral = {hmint} +- {ehmint}')
        print(f'eff = {eff}  effm = {effm}  bias eff = {effb}')


        
    except KeyboardInterrupt:
        # Gestisce l'interruzione del programma con Ctrl+C
        print("\nProgramma terminato.")


