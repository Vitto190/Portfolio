import numpy as np
import matplotlib.pyplot as plt

def montecarlo(S, mu, sigma, q, t, target, n_sim=100000, n_step=100):
    """Simulazione monte carlo di un'azione.
    nomenclatura: (tutti float):
    S  prezzo attuale dell'azione
    mu rendimento annuo atteso/deriva
    sigma volatilità annua 
    q     divident yeald annuo
    t     tempo in anni
    target   breakeven o prezzo target da superare
    si usa il moto browniano geometrico per simulare i percorsi che il prezzo dell'azione può fare. Questa funzione grafica"""
    dt= t/n_step  #calcolo il dt
    
    Z=np.random.normal(0,1,(n_sim,n_step))
    S_percorsi=np.zeros((n_sim, n_step+1))
    S_percorsi[:,0]=S
    
    for i in range(1, n_step+1):
        S_percorsi[:, i]= S_percorsi[:, i-1]*np.exp((mu-q-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*Z[:,i-1])
        
    St=S_percorsi[:,-1] #prezzo alla scadenza t
    #statistiche di probabilità
    superamento= St- target
    vincenti= superamento>0
    prob_sup=np.mean(vincenti)*100 #probabilita in percentuale 
    #superamento in media della soglia
    scenari_soprat= superamento[vincenti]
    media_superamento=np.mean(scenari_soprat) if len(scenari_soprat)>0 else 0.0
    mediana_superamento= np.median(scenari_soprat) if len(scenari_soprat)>0 else 0.0
    
    #grafici
    plt.figure(figsize=(12,5))
    #grafico traiettorie
    plt.subplot(1,2,1)
    time_grid=np.linspace(0,t,n_step +1)
    plt.plot(time_grid, S_percorsi[:100,:].T, alpha=0.3, linewidth=0.7)
    plt.axhline(target, color='red', linestyle='--', linewidth=2, label=f'Target({target:.2f}€)')
    plt.axhline(S, color='black', linestyle=':', label=f'Inizio {S:.2f}€)')
    plt.title(f"{len(S_percorsi)} simulati dell'azione")
    plt.xlabel("Tempo in anni")
    plt.ylabel("Prezzo azione €")
    plt.grid(True, alpha=0.3)
    plt.legend
    #grafico distribuzione del prezzo finale
    plt.subplot(1,2,2)
    plt.hist(St,bins=80, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(target, color='red', linestyle='--', linewidth=2, label=f'Target({target:.2f}€)')
    plt.axvline(np.mean(St), color='green', linestyle='-',label=f'Prezzo medio finale ({np.mean(St):.2f}€)')
    plt.title(f"Distribuzione prezzo finale a {t} anni\n Probabilità sopra target {prob_sup:.1f}%")
    plt.xlabel("Prezzo finale Azione St")
    plt.ylabel("Densità")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    #vari print
    print(f"Risultati simulazione montecarlo di prezzo iniziale {S} con target/breakeven {target } e orizzonte temporale di {t} anni")
    print(f"Probabilità superamento target : {prob_sup :.2f}%")
    print(f"Superamento medio oltre il target netto : {media_superamento:.2f}€")
    print(f"Superamento tipico mediana : {mediana_superamento:.2f}€")
