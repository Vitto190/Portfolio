import numpy as np
from scipy.stats import norm
def scholes(S,K,t,r, sigma, q=0.0, j=0.0, option_type='call'):
    """questa funione usa il modello black scholes che però è sistemato per dividendi e commissioni fisse (tantum da j euro).
    in questo caso i dividendi q modificano la dinamica dell'equazione differenziale di scholes classica, le comissioni 
    non modificano le curva generale ipotizzata dell'azione ma impattano il valore del contratto in sè.
    Per chi compra il prezzo calcolato con scholes aumenta di j, per chi vende diminuisce di j e il prezzo di breakeven alla 
    scadenza è K+Premio pagato dall'opzione + j (in caso di call).
    Nomenclatura:
    S float --> prezzo dell'azione
    K float     strike price di esercizio
    t float     tempo alla scadenza in anni
    r           tasso interesse annuo (0.05 per 5%)
    sigma       volatilità annua dell'azione a cui si riferisce l'opzione ( o commodity)
    q           rendimento annuo da dividendo
    j           commissione fissa
    option_type str   call o put
    analisi_opzione   ritorna il prezzo teorico puro, il prezzo di acquisto, il prezzo di vendita e breakeven"""
    #calcolo i termini che servono per l'equazione differenziale, ovvero fattori di sconto e d1/d2 (servono per calcolare la 
    #probabilità e applicare la distribuzione normale N(x) : N(d2) è la probabilità teorica in regime neutrale di rischio che alla 
    #scadenza il prezo dell'azione S sia superiore a K mentre N(d1) misura la sensibilita del prezzo dell'opzione rispetto all'
    #azione e tiene conto di quanto in alto può andare il prezzo rispetto alla sua varianza (delta dell'opzione, quante deviazioni
    #standard dista la posizione attuale dell'azione dallo strike price)). Più la volatilità è alta o il tempo è lungo più le due d 
    #si allontanano tra loro (per esempio se c'è alta volatilità la probabilità che l'azione superi K(N(d2)) si riduce rispetto al valore
    #atteso del movimento N(d1). Poi sono calcolati i fattori che identificano per il rendimento dei dividenti e il tasso di interesse)
    d1=(np.log(S/K)+(r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    d2=d1-sigma*np.sqrt(t)
    df_r=np.exp(-r*t)
    df_q=np.exp(-q*t)
    
    #calcolo il prezzo teorico
    if option_type.lower()=='call':
        prezzo_scholes=S*df_q*norm.cdf(d1)-K*df_r*norm.cdf(d2) #norm.cfd è la gaussiana standardizzata
        breakeven_expiry=K+prezzo_scholes+j
    elif option_type.lower()== 'put':
        prezzo_scholes=K*df_r*norm.cdf(-d2)-S*df_q*norm.cdf(-d1)
        breakeven_expiry= K-(prezzo_scholes+j)
    else:
        raise ValueError("option_type deve essere call o put")
    costo_acquisto= prezzo_scholes + j #commissioni
    costo_vendita= prezzo_scholes -j 
    return {
        'Tipo opzione': option_type.upper(),
        'Prezzo teorico scholes': prezzo_scholes ,
        'Costo totale acquisto': costo_acquisto ,
        'Incasso netto vendita': costo_vendita ,
        'Breakeven alla scadenza': breakeven_expiry
    }