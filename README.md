# Genova

`Genova` is a surrogate model for predicting kilonova spectra and multi-band light curve quantiles from binary neutron-star parameters.

The package includes a reference trained surrogate and supports propagation of gravitational wave posterior samples into probabilistic kilonova predictions.

## Installation

For install from source

```
git clone https://github.com/Xiaofei-D/Genova.git
cd Genova
pip install -e .
```

For development:

```
pip install -e ".[dev]"
```

Python >= 3.10 is required.

### CPU and GPU support

Genova can run on both CPU and CUDA-enabled GPUs.

By default, the device is selected automatically: a CUDA GPU is used when
available, otherwise Genova falls back to CPU.

You can check whether PyTorch can access a GPU with:

```bash
python -c "import torch;
print(torch.cuda.is_available());
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## Usage

### Command line

Predict light curves from a posterior CSV:

```
genova \
    --posterior-csv posterior.csv \
    --event GW170817 \
    --frame obs \
    --n-samples 1 \
    --bands ps1::g ps1::r ps1::i ps1::z \
    --quantiles 0.05 0.16 0.50 0.84 0.95 \
    --time-min 0.4 \
    --time-max 10.0 \
    --n-times 100 \
    --magnitudes \
    --output outputs/GW170817/lightcurve_prediction.npz
```

The CLI automatically removes posterior samples outside the supported domain of the selected surrogate before prediction, and writes the filtered samples to `<event>_posteriors_in_convex_hull.csv` alongside the prediction output.

Use

```
genova --help
```

to see all available options, including custom surrogate and device settings.

### Python

Predict spectra directly:

```
import numpy as np
from genova import load_flow_surrogate, predict_spectra

metadata, flow_model = load_flow_surrogate()

spectra = predict_spectra(
    flow_model,
    metadata,
    case_conditions=np.array([
        [1.40, 1.30, 500.0, 700.0, 0.50]
    ]),
    time_rest=np.array([1.0, 3.0, 7.0]),
    n_samples=100,
)
```

Predict light curves from a posterior. Unlike the CLI, the Python API does not filter the posterior to the surrogate's supported domain automatically, so filter it first with `posterior_in_convex_hull`:

```
from genova import EventLightcurveModel
from genova.validation import posterior_in_convex_hull

posterior_in_convex_hull("posterior.csv", "posterior_in_hull.csv")

model = EventLightcurveModel.from_csv("posterior_in_hull.csv", event="GW170817")

prediction = model.predict_lightcurve_quantiles(
    frame="obs",
)
```

To also return the individual magnitude samples:

```
prediction = model.predict_lightcurve_quantiles(
    frame="obs",
    return_magnitudes=True,
)
```

## Input

Each binary neutron-star system is described by

| Parameter       | Description                   | Unit          |
| --------------- | ----------------------------- | ------------- |
| `m1`            | primary mass                  | solar mass    |
| `m2`            | secondary mass                | solar mass    |
| `lambda1`       | primary tidal deformability   | dimensionless |
| `lambda2`       | secondary tidal deformability | dimensionless |
| `cos_theta_obs` | cosine of viewing angle       | dimensionless |

For posterior light-curve prediction, the input CSV contains

```
m1
m2
lambda1
lambda2
cos_theta_obs
redshift
luminosity_distance_Mpc
```

Each row represents one posterior sample.

Rest-frame evaluation times are specified separately in days and must lie
within the time support of the selected surrogate.

## Output

### Spectra

```
(n_cases, n_samples, n_times, n_wavelengths)
```

The spectra are rest-frame $F_\lambda$ in erg s⁻¹ cm⁻² Å⁻¹, normalised to a distance of 10 pc.

### Light curves

`predict_lightcurve_quantiles` returns a dictionary whose main keys are

| Key                                | Shape                                   |
| ---------------------------------- | --------------------------------------- |
| `time_obs` or `time_rest_lc`       | `(n_times,)`, depending on `frame`      |
| `magnitude_quantiles`              | `(n_quantiles, n_times, n_bands)`       |
| `valid_draws`                      | `(n_times, n_bands)`                    |
| `magnitude_samples` (optional)     | `(n_cases, n_samples, n_times, n_bands)` |

Photometry is reported in AB magnitudes.

For command-line predictions, outputs are saved as compressed NPZ files.

## Conventions

- Masses are in solar masses.
- Tidal deformabilities are dimensionless.
- `cos_theta_obs = cos(theta_obs)` and lies in `[0, 1]`.
- Time is in days.
- Wavelength is in Angstrom.
- Spectra are rest-frame $F_\lambda$ at 10 pc.
- `frame="obs"` gives observer-frame apparent magnitudes using the supplied redshift and luminosity distance.
- `frame="rest"` gives absolute magnitudes at 10 pc.
- Light-curve quantiles are computed from magnitude samples.
- The CLI restricts predictions to the supported domain of the selected
  surrogate; in Python, use `posterior_in_convex_hull` before predicting.

## Reference surrogate

A reference N21 surrogate is bundled with Genova. It assumes `m1 >= m2` and `lambda1 <= lambda2`.

The reference `N21` surrogate is trained on kilonova spectra generated using the corresponding implementation in [Redback](https://redback.readthedocs.io/en/latest/api/redback.transient_models.kilonova_models.nicholl_bns_rk24.html), which is adapted from the forward model of [Nicholl et al. (2021)](https://doi.org/10.1093/mnras/stab1523)


Further details on its training domain and supported parameter space are given in `docs/surrogate_n21.md`.

## Citation

If you use Genova in your research, please cite:

Dong et al. (2026), [arXiv:2608.20262](https://arxiv.org/abs/2608.20262)