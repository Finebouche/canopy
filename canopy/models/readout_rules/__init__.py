"""
Readouts trained online: least mean squares (lms) and exploratory Hebbian learning (eh) in JAX (the steps of a pass
over the rows in a compiled lax.scan), and the exploratory Hebbian rule as a NumPy reservoirpy node (eh_numpy, the
reference of eh: the same weights); their versions on the monomials of the state, in dual form (kernel); recursive
least squares (rls: reservoirpy's node with the bias as the weight of a constant input).

Which version is which (the rules of canopy.next_phrase_prediction.readout_rules), detailed in the docstrings of lms
and eh_numpy:
  - "rls": RLS (the RLS-based FORCE rule of Sussillo & Abbott 2009), the ridge regression after one pass;
  - "lms": LMS (Widrow & Hoff 1960), a constant learning rate (reservoirpy's LMS);
  - "nlms": NLMS (Nagumo & Noda 1967, Albert & Gardner 1967), the learning rate divided by ||x||^2 + 1 at each row;
  - "eh": the exploratory Hebbian rule of Hoerzer, Legenstein & Maass (2014);
  - "eh_np", "eh_np_local", "neh": ours, EH with the noise known (node perturbation, Fiete & Seung 2006), then a
    reward per output, then the NLMS step (closest published: Burms, Caluwaerts & Dambre 2015);
  - "nlms:<n>", "neh:<n>": nlms and neh on the monomials of Ohkubo & Inubushi (2024) in dual form (kernel LMS, Liu,
    Pokharel & Principe 2008, normalized): ours.
"""
