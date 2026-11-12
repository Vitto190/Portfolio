#SENZA YFINANCE
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# 1. MODELLO BLACK-SCHOLES-MERTON (BSM)

def scholes(S, K, t, r, sigma, q=0.0, j=0.0, option_type='call'):
    """
    Calcola il prezzo teorico, costi di transazione e breakeven.
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    df_r = np.exp(-r * t)
    df_q = np.exp(-q * t)
    
    if option_type.lower() == 'call':
        prezzo_scholes = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
        breakeven_expiry = K + prezzo_scholes + j
    elif option_type.lower() == 'put':
        prezzo_scholes = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
        breakeven_expiry = K - (prezzo_scholes + j)
    else:
        raise ValueError("option_type deve essere 'call' o 'put'")
        
    costo_acquisto = prezzo_scholes + j
    costo_vendita = prezzo_scholes - j 
    
    return {
        'Tipo opzione': option_type.upper(),
        'Prezzo teorico scholes': prezzo_scholes,
        'Costo totale acquisto': costo_acquisto,
        'Incasso netto vendita': costo_vendita,
        'Breakeven alla scadenza': breakeven_expiry,
        'K': K,
        'S': S
    }


# 2. SIMULAZIONE MONTE CARLO STANDARD (GBM)

def montecarlo(S, mu, sigma, q, t, target, n_sim=100000, n_step=100):
    """
    Simula le traiettorie e restituisce la distribuzione finale.
    """
    dt = t / n_step
    Z = np.random.normal(0, 1, (n_sim, n_step))
    S_percorsi = np.zeros((n_sim, n_step + 1))
    S_percorsi[:, 0] = S
    
    for i in range(1, n_step + 1):
        S_percorsi[:, i] = S_percorsi[:, i-1] * np.exp((mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i-1])
        
    St = S_percorsi[:, -1]
    superamento = St - target
    vincenti = superamento > 0
    prob_sup = np.mean(vincenti) * 100
    
    scenari_soprat = superamento[vincenti]
    media_superamento = np.mean(scenari_soprat) if len(scenari_soprat) > 0 else 0.0
    mediana_superamento = np.median(scenari_soprat) if len(scenari_soprat) > 0 else 0.0
    
    return {
        'Probabilita superamento': prob_sup,
        'Superamento medio': media_superamento,
        'Superamento mediana': mediana_superamento,
        'Prezzo medio finale': np.mean(St),
        'St': St
    }

# 3. MODELLO HESTON (VOLATILITÀ STOCASTICA)

def heston(S, sigma_live, mu, q, t, target, kappa, xi, rho, n_sim=100000, n_step=100):
    """
    Simulazione con modello a volatilità stocastica di Heston.
    """
    v0 = sigma_live**2
    theta = v0
    dt = t / n_step
    
    S_path = np.zeros((n_sim, n_step + 1))
    v_path = np.zeros((n_sim, n_step + 1))
    S_path[:, 0] = S
    v_path[:, 0] = v0
    
    for step in range(1, n_step + 1):
        Z1 = np.random.normal(0, 1, n_sim)
        Z2 = np.random.normal(0, 1, n_sim)
        ZS = Z1
        Zv = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        v_prev = np.maximum(v_path[:, step - 1], 0)
        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * Zv
        v_path[:, step] = np.maximum(v_path[:, step - 1] + dv, 0)
        
        S_path[:, step] = S_path[:, step - 1] * np.exp((mu - q - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * ZS)
        
    St = S_path[:, -1]
    Prob_sup = np.mean(St > target) * 100
    
    return {
        'S': S,
        'sigma_live': sigma_live,
        'target': target,
        'Prob_sup': Prob_sup,
        'St': St,
        'S_path': S_path,
        't': t
    }


def valuta_trade(res_scholes, res_mc, res_heston, option_type='call'):
    """
    Genera i 3 grafici affiancati ed eroga il report operativo.
    """
    option_type = option_type.lower()
    S_spot = res_heston['S']
    costo_totale = res_scholes['Costo totale acquisto']
    breakeven = res_scholes['Breakeven alla scadenza']
    K = res_scholes['K']
    
    pop_mc = res_mc['Probabilita superamento']
    pop_heston = res_heston['Prob_sup']
    St_heston = res_heston['St']
    t = res_heston['t']
    
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"ANALISI TRADE ({option_type.upper()}) - SPOT: {S_spot:.2f}€", fontsize=14, fontweight='bold')
    
    # Grafico 1: Payoff & P&L Black-Scholes
    ax1 = fig.add_subplot(2, 2, 1)
    S_range = np.linspace(S_spot * 0.8, S_spot * 1.2, 200)
    payoff = np.maximum(S_range - K, 0) if option_type == 'call' else np.maximum(K - S_range, 0)
    pnl = payoff - costo_totale
    
    ax1.plot(S_range, pnl, label='P&L Netto', color='blue', linewidth=2)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(S_spot, color='gray', linestyle=':', label=f'Spot ({S_spot:.2f}€)')
    ax1.axvline(K, color='orange', linestyle='--', label=f'Strike K ({K:.2f}€)')
    ax1.axvline(breakeven, color='red', linestyle='-', label=f'Breakeven ({breakeven:.2f}€)')
    ax1.fill_between(S_range, pnl, 0, where=(pnl > 0), color='green', alpha=0.2)
    ax1.fill_between(S_range, pnl, 0, where=(pnl < 0), color='red', alpha=0.2)
    ax1.set_title("1. Black-Scholes: Profilo P&L alla Scadenza")
    ax1.set_xlabel("Prezzo Azione St (€)")
    ax1.set_ylabel("Profitto / Perdita (€)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Grafico 2: Distribuzione Monte Carlo Standard
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(res_mc['St'], bins=70, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.axvline(breakeven, color='red', linestyle='--', linewidth=2, label=f'Target BE ({breakeven:.2f}€)')
    ax2.axvline(np.mean(res_mc['St']), color='green', linestyle='-', label=f'Media ({np.mean(res_mc["St"]):.2f}€)')
    ax2.set_title(f"2. Monte Carlo (GBM): Distribuzione Prezzo Finale\nPoP = {pop_mc:.1f}%")
    ax2.set_xlabel("Prezzo Finale St (€)")
    ax2.set_ylabel("Densità")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Grafico 3: Traiettorie Heston (Volatilità Stocastica)
    ax3 = fig.add_subplot(2, 1, 2)
    time_grid = np.linspace(0, t, res_heston['S_path'].shape[1])
    ax3.plot(time_grid, res_heston['S_path'][:80, :].T, alpha=0.3, linewidth=0.7)
    ax3.axhline(breakeven, color='red', linestyle='--', linewidth=2, label=f'Breakeven ({breakeven:.2f}€)')
    ax3.axhline(S_spot, color='black', linestyle=':', label=f'Spot iniziale ({S_spot:.2f}€)')
    ax3.set_title(f"3. Heston: Percorsi di Prezzo (IV Inserita: {res_heston['sigma_live']*100:.1f}% | PoP: {pop_heston:.1f}%)")
    ax3.set_xlabel("Tempo (Anni)")
    ax3.set_ylabel("Prezzo (€)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()

   
    payoff_heston = np.maximum(St_heston - K, 0) if option_type == 'call' else np.maximum(K - St_heston, 0)
    ev_heston = np.mean(payoff_heston - costo_totale)
    distanza_be_pct = (breakeven - S_spot) / S_spot * 100 if option_type == 'call' else (S_spot - breakeven) / S_spot * 100


    print(f"Prezzo Spot Azione (S)       : {S_spot:.2f} €")
    print(f"Strike Price (K)             : {K:.2f} €")
    print(f"Costo Totale Opzione + Comm. : {costo_totale:.2f} €")
    print(f"Breakeven Scadenza           : {breakeven:.2f} € (Distanza: {distanza_be_pct:+.2f}%)")
    
    print(f" MONTE CARLO STANDARD (GBM): PoP = {pop_mc:.1f}%")
    print(f" HESTON (Vol. Stocastica)  : PoP = {pop_heston:.1f}%  | Expected Value = {ev_heston:+.2f} €")
    
    
    
    if ev_heston > 0 and pop_heston >= 45:
        print(" ACQUISTO CONSIGLIATO (Long Strategy):")
        print("   L'operazione ha un'aspettativa matematica NETTA POSITIVA sotto Heston.")
        print("   La deriva e la volatilità coprono le commissioni e il premio dell'opzione.")
        
    elif ev_heston > 0 and pop_heston < 45:
        print(" ACQUISTO SPECULATIVO / ASIMMETRICO:")
        print("   La probabilità di vincita è sotto il 45%, ma l'esplosività dei guadagni nei casi")
        print("   favorevoli mantiene il valore atteso positivo. Ridurre il dimensionamento (Size).")
        
    else:
        print(" ACQUISTO SFAVOREVOLE (Evitare acquisto diretto):")
        print("   L'aspettativa matematica è negativa. Il breakeven è troppo penalizzante o")
        print("   il premio/commissioni appesantiscono troppo la posizione.")
        
        if pop_heston < 35:
            print("\n    ALTERNATIVA OPERATIVA (Vendita / Credit Strategy):")
            print("      Con una probabilità di superare il breakeven inferiore al 35%, il margine")
            print("      statistico passa dalla parte del VENDITORE dell'opzione.")

    diff_pop = pop_heston - pop_mc
    if abs(diff_pop) >= 2.0:
        if diff_pop > 0:
            print(f" La volatilità stocastica aumenta il PoP del +{diff_pop:.1f}%.")
            print("   L'effetto 'fat tails' rende più probabili i movimenti estesi a tuo favore.")
        else:
            print(f" La volatilità stocastica riduce il PoP del {diff_pop:.1f}%.")
            print("   La riconvergenza della volatilità (kappa) tende a limitare i rally estremi.")
    print("=" * 65)
    return
