from methods import baseline, ewc, mas, icarl, fedavg, fedprox, fedcurv, fedweit, fedstil, fedstil_atten

methods = {
    'baseline': baseline,
    'ewc': ewc,
    'mas': mas,
    'icarl': icarl,
    'fedavg': fedavg,
    'fedprox': fedprox,
    'fedcurv': fedcurv,
    'fedweit': fedweit,
    'fedstil': fedstil,
    'fedstil-atten': fedstil_atten,
}
