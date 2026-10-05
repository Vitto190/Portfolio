import numpy as np
import yfinance as yf 
import matplotlib.pyplot as plt
#yfinance scarica i dati di volatilità reali (Vix o volatilità ATM di una singola azione) da Yahoo Finance

def volatilità(ticker_symbol):
    """Scarica il prezzo e la volatilità ATM dell'azione tramite yfinance. Se non sono disponibili, usa il VIX. Si imposta il prezzo
    target a +10% di S di base, basta cambiarlo
    """
    tk =yf.Ticker(ticker_symbol)
    
    #prezzo azione
    data=tk.history(period="1d") #1 giorno di periodo
    if data.empty:
        raise ValueError(f"Ticker '{ticker_symbol}' non trovato")
    P= data['Close'].iloc[-1]
    sigma_ATM= None
    
    #estriamo la volatilità implicita ATM dalla catena delle opzioni
    try:
        expirations=tk.options
        if len(expirations)>0:
            #uso la prima scadenza dispinibile
            opt_chain=tk.option_chain(expirations[0])
            calls=opt_chain.calls
            
            #troviamo lo strike più vicino allo Spot S (ATM)
            #NB lo strike prize è è il prezzo al quale è possibile esercitare la “facoltà” associata ad un’opzione da parte del suo possessore.
            idx_atm=(calls['strike']-P).abs().idxmin()
            sigma_ATM=calls.loc[idx_atm, 'impliedVolatility']
            print(f"Volatilità Implicita ATM estratta per {ticker_symbol}: {sigma_ATM*100:.2f}%")
    except Exception as e:
        print(f"Impossibile recuperare catena opzioni per {ticker_symbol}:{e}")
    #Uso VIX se l'azione non ha catene di opzioni disponibili su yahoo finance
    if sigma_ATM is None or np.isnan(sigma_ATM) or sigma_ATM==0:
        vix=yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1]
        sigma_ATM=vix/100.0
        print(f"Usando il Vix come stima di volatilità di mercato {sigma_ATM*100:.2f}%")
    return P, sigma_ATM

def heston(ticker, mu, q, t, kappa, xi, rho, n_sim=10000, n_step=100):
    #calibra automaticamente v0 e theta di Heston in base alla volatilità di mercato in tempo reale
    #uso la condizione di feller (2ktheta>xi**2) per evitare che la varianza diventi negativa durante la simulazione
    #nel codice uso  la tecnica Full Truncation (max(vt,0))
    #mu=rendimento atteso annuo dell'azione
    #q= divident yeald annuo
    #t tempo in anni
    #kappa= velocità di riconvergenza (velocità con cui la volatilità torna verso il suo valore medio di lungo termine theta)
    #xi=volatilità della varianza (quanto è nervosa la volatilità e quanti ampi sono gli sbalzi casuali)
    #rho= correlazione prezzo-volatilità: legame stocastico tra prezzo dell'azione e varianza (se negativo i crolli di prezzo
    #corrispondo a impennate nella volatilità)
    #
    #scaricare i dati live
    S, sigma_live=volatilità(ticker)
    #parametri
    v0=sigma_live**2 #varianca iniziale basata sulla IV attuale
    theta=v0 #assumiamo che la varianza di lungo termine corrisponda al livello attuale (che sia a sua volta quella di medio termine)
    target= S *1.10 #imposto il target a +10% dello Spot attuale
    dt= t/n_step
    
    S_path= np.zeros((n_sim, n_step+1))
    v_path=np.zeros((n_sim,n_step+1))
    S_path[:,0]=S
    v_path[:,0]=v0
    #simulazione Heston
    for step in range(1, n_step+1):
        Z1=np.random.normal(0,1,n_sim)
        Z2=np.random.normal(0,1,n_sim)
        ZS=Z1
        Zv=rho*Z1+np.sqrt(1-rho**2)*Z2
        v_prev=np.maximum(v_path[:,step-1],0)
        
        dv=kappa*(theta-v_prev)*dt +xi*np.sqrt(v_prev*dt)*Zv
        v_path[:,step]=np.maximum(v_path[:,step-1]+dv,0)
        
        S_path[:,step]=S_path[:,step-1]*np.exp((mu-q-0.5*v_prev)*dt+np.sqrt(v_prev*dt)*ZS)
    St=S_path[:,-1]
    Prob_sup=np.mean(St>target)*100
    #Risultati
    print(f"Prezzo Spot iniziale S :{S:.2f} €")
    print(f"Volatilità Implicita Calibrata :{sigma_live*100:.2f} %")
    print(f"Soglia target (+10% dal prezzo) :{target:.2f} €")
    print(f"Probabilità di superamento :{Prob_sup:.2f} %")
    #grafici
    time_grid=np.linspace(0,t,n_step+1)
    plt.figure(figsize=(10,4))
    plt.plot(time_grid, S_path[:80,:].T, alpha=0.3, linewidth=0.7)
    plt.axhline(target, color='red', linestyle='--', label=f'Target ({target:.2f})')
    plt.axhline(S, color='black', linestyle=':', label=f'Spot({S:.2f})')
    plt.title(f"Traiettorie Heston calibrato su IV live di {ticker} ({sigma_live*100:.1f}%)")
    plt.xlabel("Anni")
    plt.ylabel("Prezzo")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    return {
    'S': S,
    'sigma_live': sigma_live,
    'target': target,
    'Prob_sup': Prob_sup,
    'St': St
    }
