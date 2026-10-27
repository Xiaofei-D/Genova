# The N21 reference surrogate

The N21 surrogate emulates rest-frame kilonova spectra from the Redback `nicholl_bns_rk24` model, the Nicholl et al. (2021) binary neutron-star model with the corrected prompt-collapse threshold.

Files: `src/genova/models/reference_surrogates/N21_{flow.pt, flow_config.json, convex_hull.npz}`

## Inputs

The six conditions of the surrogate. Limits show the actual marginal ranges in the final training set.

| Condition | Min | Max | Unit |
| --- | --- | --- | --- |
| `time_rest` | 0.1 | 11.0 | days (log-uniform) |
| `m1` | 1.100 | 1.800 | solar masses |
| `m2` | 1.100 | 1.798 | solar masses |
| `lambda1` | 53.3 | 2999 | dimensionless |
| `lambda2` | 55.9 | 3000 | dimensionless |
| `cos_theta_obs` | 0.0 | 1.0 | dimensionless (uniform in cosine) |

## Supported domain

The bundled convex hull, `N21_convex_hull.npz`, is the binary-parameter domain filter used by Genova. It is constructed in `(m1, m2, log10(lambda1), log10(lambda2))`.

The `genova` command applies this filter automatically, but direct Python use does not: filter inputs explicitly with `genova.validation.posterior_in_convex_hull` before predicting.

## Conventions and training-data ranges

Relations and ranges represented in the final training data. The surrogate carries no information about binaries outside them.

| Quantity | Range or relation |
| --- | --- |
| Component ordering | `m1 >= m2`, `lambda1 <= lambda2` |
| Mass ratio `q = m2/m1` | 0.611–1.0 |
| Compactness | 0.121–0.230, both components |
| Radius | 8.5–15.0 km, both components |

## Nuisance parameters

Varied when generating the training spectra but not exposed as conditions. The surrogate marginalises over them, so its predictive spread includes their contribution across these ranges.

| Parameter | Min | Max | Meaning |
| --- | --- | --- | --- |
| `kappa_red` | 1.0 | 30.0 | grey opacity of the red component, $\mathrm{cm}^{-2}$ $\mathrm{g}^{-1}$ |
| `kappa_blue` | 0.01 | 1.0 | grey opacity of the blue component, $\mathrm{cm}^{-2}$ $\mathrm{g}^{-1}$ |
| `mtov` | 2.0 | ≤ 2.5 | maximum non-rotating neutron star mass, solar masses; upper limit depends on the binary by the prompt-collapse consistency selection |
| `epsilon` | 0.1 | 0.5 | fraction of the disk mass ejected |
| `alpha` | 0.1 | 1.0 | enhancement of the blue ejecta by neutron star surface winds |
| `cos_theta_open` | 0.5 | 0.866 | Cosine of lanthanide-rich half-opening angle |
| `cos_theta_cocoon` | 0.866 | 1.0 | Cosine of shocked-cocoon opening angle |
| `Tf1` | 1000 | 4000 | temperature floor of component 1, K |
| `Tf2` | 1000 | 4000 | temperature floor of component 2, K |
| `Tf3` | 1000 | 4000 | temperature floor of component 3, K |

## Output

The direct output contains 20 wavelength bins over 2800–25000 Å.
Fluxes are rest-frame $F_\lambda$ at 10 pc, with units of
`erg s^-1 cm^-2 Å^-1`.

`predict_spectra` returns an array of shape `(n_cases, n_samples, n_times, 20)`.

The flow models the min-max scaled $y = (x - x_\mathrm{min}) / (x_\mathrm{max} - x_\mathrm{min})$
of $x = \mathrm{asinh}(F_\lambda / s)$, with $s$ = $10^{-7}$
`erg s^-1 cm^-2 Å^-1` representing `asinh_scale`. $s$, `x_min` and `x_max` are stored in `N21_flow_config.json`.
Predictions are returned as $F_\lambda = s\,\sinh(x_\mathrm{min} + y\,(x_\mathrm{max} - x_\mathrm{min}))$,
with negative values clipped to zero.
