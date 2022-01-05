# Materials & Methods

## Data processing
The dataset exported from the laboratory automation platform was processed into a set of tabular `DataFrame` structures using `pandas` [pandas].
Every unique combination of glucose feed rate and IPTG concentration was assigned a unique identifier (`design_id`) for identification inside the model.
Likewise, every biotransformation reaction was assigend a `replicate_id`.
The association between all experiment designs, biotransformation reactions and relevant metainformation such as assay well positions was tracked in a tabular form ("df_layout" sheet in `dataset.xlsx`).
Reference wells of known product concentration were equally included in the dataset, hence the layout table includes a sparse column for known product concentrations.

Measurements of absorbance at 360 and 600&nbsp;nm were were kept in separate tables ("df_360" and "df_600" in `dataset.xlsx`), organized by the previously assigned `replicate_id`.

## Calibration models
### Biomass concentration
The separately acquired biomass calibration dataset was used to fit two models describing the relationship between biomass cell dry weight and absorbance at 360 and 600&nbsp;nm respectively.

![](figures/calibration_biomass360.png)

__Figure 1: Biomass calibration at 360 nm.__ The spread of observations (🔵) is modeled by a `calibr8.LogIndependentAsymmetricLogisticT` model with `scale_degree=1` to account for non-linearity (left) and heteroscedasticity (right). Green bands depict the intervals of 97.5, 95 and 84&nbsp;% probability of observations according to the model.

![](figures/calibration_biomass600.png)

__Figure 2: Biomass calibration at 600 nm.__ Observations (🔵) at 600&nbsp;nm indicated lower absorbance compared to 360&nbsp;nm. Like for 360&nbsp;nm, the model is a `calibr8.LogIndependentAsymmetricLogisticT` model with `scale_degree=1`.

The models were built with the `calibr8` package [calibr8] using an asymmetric logistic function of the logarithmic biomass concentration to describe the mean of Students-*t* distributed absorbance observations.
Since the absorbance/biomass relationship exhibits a heteroscedastic noise, the scale parameter of the Students-*t* distribution was modeled as linearly dependent on the mean.
The degree of freedom parameter $\nu$ was estimated as a constant.

### Product concentration
The ABAO reaction was performed to quantify 2-amino benzaldehyde.
The absorbance of its reaction product was measured at 360&nbsp;nm in all assays.
A separate calibration dataset was obtained by performing the assay procedure on reference samples with known 2-amino benzaldehyde concentrations.
Reference samples were prepared without biomass and with different amounts of acetic acid to exclude biomass absorbance, and investigate pH robustness of the method.

A linear calibration model with scale and $\nu$ parameters of the Students-*t* distribution as constants was fitted to the 360&nbsp;nm measurements of product calibration samples.

![](figures/calibration_product360.png)

__Figure 3: Product calibration at 360 nm.__ In the observed range, the absorbances at 360&nbsp;nm (🔵) followed a linear trend in dependence on the 2-amino benzaldehyde concentration. The model was built from a `calibr8.BasePolynomialModelT` model with `mu_degree=1` and `scale_degree=0`.

All calibration model parameters were estimated by maximum likelihood using SciPy optimizers.
For code and reproducable Jupyter notebooks of this analysis we refer to the accompanying GitHub repository.

## Process model
The dataset of biotransformation reactions was modeled in a generative hierarchical Bayesian framework using the probabilistic programming language PyMC [pymc3,pymcZenodo].
This model closely resembles the biotechnological process that generated the dataset, therefore we call it *process model* henceforth.
Starting from input parameters such as specific activity, random effects or dependence of final 10&nbsp;mL reactor biomass concentration on glucose feed rate the process model simulates biomass and 2-amino benzaldehyde concentrations in each biotransformation well across all experiments.

A likelihood needed for parameter inference by Markov-chain Monte Carlo (MCMC) is created from process model predictions and observed absorbances according to relationships described by the separately fitted calibration models $\phi_\mathrm{cm,X,600\ nm}$, $\phi_\mathrm{cm,X,360\ nm}$ and $\phi_\mathrm{cm,P,360\ nm}$.
At 600&nbsp;nm this is the likelihood of the observed data given the predicted biomass concentration $X$.
At 360&nbsp;nm however, both biomass $X$ and ABAO reaction product absorb and therefore the sum of their absorbances needs to be taken into account for the likelihood.

Note that while it is the ABAO reaction product that contributes absorbance at 360&nbsp;nm we performed the ABAO assay calibration with known 2-amino benzaldehyde concentrations, so the corresponding model $\phi_\mathrm{cm,P,360\ nm}$ describes 360&nbsp;nm ABAO reaction product absorbance as a function of 2-amino benzaldehyde concentration.
For simplicity we therefore use the symbol $P$ to refer to the product of interest concentration: 2-amino benzaldehyde in the biotransformation solution.

$$
\begin{aligned}
    \mathcal{L_\Sigma} &= \mathcal{L}_\mathrm{600\ nm}(\mathrm{A_{600\ nm}} \mid \mathrm{A_{600\ nm,obs}}) \cdot \mathcal{L}_\mathrm{360\ nm}(\mathrm{A_{360\ nm}} \mid \mathrm{A_{360\ nm,obs}}) \\
    \textrm{where} \\
    \mathrm{A_{600\ nm}} &\sim Normal(\mathrm{\mu_{X,600\ nm}}, \mathrm{\sigma_{X,600\ nm}}) \\
    (\mathrm{\mu_{X,600\ nm}}, \mathrm{\sigma_{X,600\ nm}}) &= \phi_\mathrm{X,600\ nm}(X) \\
    \mathrm{A_{360\ nm}} &\sim Normal(\mathrm{\mu_{360\ nm}}, \mathrm{\sigma_{360\ nm}}) \\
    \mathrm{\mu_{360\ nm}} &= \mathrm{\mu_{X,360\ nm}} + \mathrm{\mu_{P,360\ nm}} \\
    \mathrm{\sigma_{360\ nm}} &= \sqrt{\mathrm{\sigma_{X,360\ nm}}^2 + \mathrm{\sigma_{P,360\ nm}}^2} \\
    (\mathrm{\mu_{X,360\ nm}}, \mathrm{\sigma_{X,360\ nm}}) &= \phi_\mathrm{cm,X,360\ nm}(\mathrm{\vec{X}_{\vec{t},\vec{replicate}}}) \\
    (\mathrm{\mu_{X,360\ nm}}, \mathrm{\sigma_{X,360\ nm}}) &= \phi_\mathrm{cm,P,360\ nm}(\mathrm{\vec{P}_{\vec{t},\vec{replicate}}}) \\
\end{aligned}
$$

The above observation model applies to biomass $X$ and 2-amino benzaldehyde concentration $P$ _at every time point_, _in every replicate_ of either a biotransformation reaction or reference sample.
Reference wells of known product concentrations, but without 2-amino benzoic acid are also included in the model, albeit with the assumption that the 2-amino benzaldehyde concentration remains constant over time.

$$
\begin{aligned}
    \mathrm{\vec{X}_{\vec{t},\vec{replicate}}} &= \{ \mathrm{\vec{X}_{\vec{t},\vec{reference}}}, \mathrm{\vec{X}_{\vec{t},\vec{DWP}}}\} \\
    \mathrm{\vec{P}_{\vec{t},\vec{replicate}}} &= \{ \mathrm{\vec{P}_{(\vec{t}),\vec{reference}}}, \mathrm{\vec{P}_{\vec{t},\vec{DWP}}}\} \\
\end{aligned}
$$

The process model to describe these _per replicate_ and _per time point_ concentrations is described in the following sections.

Since almost all process model variables are vectors or matrices, we denote dimensions by subscripts with arrows.
For example, the notation $\vec{X}_{\vec{t},\vec{DWP}}$ or $\vec{P}_{\vec{t},\vec{DWP}}$ should be interpreted as 2-dimensional variables (matrices) with entries for each combination of time point and DWP well.
The meanings of dimension symbols is summarized in the following table.

| symbol | dimension length | variable has elements for each of the... |
| - | - | - |
| $\vec{DASGIP}$ | 4 | DASGIP reactor batches. |
| $\vec{2mag}$ | 191 | 2mag reactor vessels. |
| $\vec{DWP}$ | 191 | DWP wells with active biotransformations. |
| $\vec{replicate}$ | 263 | DWP wells, which includes biotransformation and reference wells. |
| $\vec{t}$ | 5 | time points at which observations were made. |
| $\vec{glc}$ | 6 | glucose feed rates investigated. |
| $\vec{IPTG}$ | 25 | IPTG concentrations investigated. |
| $\vec{design}$ | 42 | unique combinations of glucose feed rate & IPTG concentration. |


### Biomass process model
The biomass in the experiment is sourced from a "seed train" of cultivations in three different scales and operating modes:
1. 1&nbsp;L DASGIP batch bioreactor; 1 per experiment run.
2. 10&nbsp;mL 2mag fed-batch macrobioreactor; 48 per experiment run.
3. 600&nbsp;µL biotransformation in square deep-well plate; 60 per experiment run.

The process model must describe biomass in each biotransformation well so it can be accounted for in the 360&nbsp;nm absorbance.
Since a universally activity metric, that can be interpredeted independent from experimenteal batch effects is desired, the model must additionally describe biomass in a way that excludes random batch effects.
The first process stage at which such an experiment-independent prediction is needed, is the final biomass concentration of the 1&nbsp;L batch cultivation.

Concretely, we describe the per-experiment final biomass concentration at the 1&nbsp;L scale as a LogNormal-distributed variable called $\mathrm{\vec{X}_{end,\vec{DASGIP}}}$ with an entry for each experiment run.
To obtain an experiment-independent prediction, we introduced $\mathrm{X_{end,batch}}$ as a _group mean prior_, also known as a _hyperprior_, around which the $\mathrm{\vec{X}_{end,\vec{DASGIP}}}$ is centered.
The prior on $\mathrm{X_{end,batch}}$ is weakly (large $\sigma$) centered at $0.5\ g/L$, whereas actual batches should only deviate from that group mean by ca. $5\ \%$.

$$\begin{aligned}
    \mathrm{X_{end,batch}} &\sim LogNormal(\mu=ln(0.5), \sigma=0.5)\\
    \mathrm{\vec{X}_{end,\vec{DASGIP}}} &\sim LogNormal(\mu=ln(\mathrm{X_{end,batch}}), \sigma=0.05)
\end{aligned}$$

This hierarchical structure is a common motif in Bayesian modeling since it enables a model to learn variables that are essential to the process understanding (here: $\mathrm{X_{end,batch}}$) while retaining the ability to describe the fine-grained structure of the experimental data (here: $\mathrm{\vec{X}_{end,\vec{DASGIP}}}$).
The motif of hierarchically modeled variables was used in several places of our bioprocess model.
For a thorough introduction to hierarchical modeling, we recommend [betancourt2020].

The second process stage in the biomass seed train is the expression in 10&nbsp;mL scale under fed-batch conditions.
Every 10&nbsp;mL 2mag reactor was inoculated with culture broth from a DASGIP reactor, hence a mapping $f_\mathrm{\vec{DASGIP} \rightarrow \vec{2mag}}$ yields initial biomass concentrations $\mathrm{\vec{X}_{start,\vec{2mag}}}$ by sub-indexing the $\mathrm{\vec{X}_{end,\vec{DASGIP}}}$ variable.
The experimental design of the fed-batches comprised varying glucose feed rates and IPTG concentrations.
It is plausible to assume a dependence of the final biomass concentration $\mathrm{\vec{X}_{end,\vec{2mag}}}$ on the glucose feed rate.
Without any mechanistic assumptions, we lump the final biomass concentration per 1&nbsp;mL reactor as the product of initial biomass concentration with a positive factor $\mathrm{\vec{X}_{factor,\vec{glc}}}$ that depends on the glucose feed rate.
Dependence of $\mathrm{\vec{X}_{factor,\vec{glc}}}$ on the glucose feed rate is modeled by a Gaussian process such that our model can also interpolate and make predictions for new glucose feed rate settings.

$$
\begin{aligned}
    \mathrm{\vec{X}_{start,\vec{2mag}}} &= f_\mathrm{\vec{DASGIP} \rightarrow \vec{2mag}}(\mathrm{\vec{X}_{end,\vec{DASGIP}}}) \\
    \mathrm{\vec{X}_{end,\vec{2mag}}} &= \mathrm{\vec{X}_{start,\vec{2mag}}} \cdot f_{\mathrm{\vec{glc} \rightarrow \vec{2mag}}}(\mathrm{\vec{X}_{factor,\vec{glc}}}) \\
    \textrm{with} \\
    ln(\mathrm{\vec{X}_{factor,\vec{glc}}}) &= f_\mathrm{\vec{lnX}_{factor,\vec{glc}}}(log_{10}(\vec{D}_\mathrm{design,\vec{glc}})) \\
    f_\mathrm{\vec{lnX}_{factor,\vec{glc}}}(d) &\sim GP(0, k(d,d')) \\
    k(d,d') &= \sigma^2 \cdot e^{-\frac{(d-d')^2}{2\ell^2}} \\
    \sigma &\sim LogNormal(ln(0.3), 0.1) \\
    \ell &\sim LogNormal(ln(0.34), 0.1)
\end{aligned}
$$

The Gaussian process was parametrized with a mean function of $0$, thereby centering the prior for $\mathrm{\vec{X}_{factor,\vec{glc}}}$ around $1$.
For the covariance function we chose a scaling parameter $\sigma$ such that the prior variance for the factor is around $\pm30\ \%$.
The prior for $\ell$ in the exponential quadratic kernel encodes a belief that $\mathrm{\vec{X}_{factor,\vec{glc}}}$ varies smoothly on a length scale of around half of the (logarithmic) design space.

The 3rd and final process stage is the biotransformation.
Here, the initial biomass concentration in every DWP replicate well $\mathrm{\vec{X}_{0,\vec{replicate}}}$ equals the final biomass concentration from a corresponding 10&nbsp;mL reactor.
The biomass concentration continued to change over the course of the biotransformation, because the solution also contained glucose as a carbon source.
Inspired by the $\mu(t)$ method described in [blet] we account for this biomass growth during the biotransformation with a Gaussian random walk of the discretized growth rate $\vec{\mu}_{\vec{t},\vec{replicate}}$.
The result are biomass concentrations for every replicate well and measurement cycle $\mathrm{\vec{X}_{\vec{t},\vec{replicate}}}$.

$$
\begin{aligned}
    \mathrm{\vec{X}_{0,\vec{replicate}}} &= f_\mathrm{\vec{2mag}\rightarrow \vec{replicate}}(\mathrm{\vec{X}_{end,\vec{2mag}}}) \\
    \mathrm{\vec{X}_{t \ge 1,\vec{replicate}}} &= \mathrm{\vec{X}_{0,\vec{replicate}}} \cdot e^{cumsum(\vec{\mu}_{\vec{t},\vec{replicate}} \cdot \vec{dt}_{\vec{t},\vec{replicate}})}\\
    \vec{\mu}_{\vec{t},\vec{replicate}} &\sim GaussianRandomWalk(\sigma=0.1) \\
\end{aligned}
$$

### Biotransformation reaction process model
Next to biomass, the second important contributor to observed absorbances is the 2-amino benzaldehyde concentration that reacted with ABAO.
This 2-amino benzalhedyde is the product of the biotransformation reaction occurrring in the DWP.
In the reference samples this concentration is known and assumed constant over time.
In the wells with 2-amino benzoic acid substrate and a $P_0=0$ initial product concentration, we model the biotransformation reaction as a 1st order reaction with an initial rate coefficient $\vec{k}_{0,\vec{DWP}}$.

$$
\begin{aligned}    
    \vec{P}_{\vec{t},\vec{DWP}} &= S_0 \cdot (1 - e^{-\vec{t}_\mathrm{actual,\vec{DWP}} \cdot \vec{k}_{0,\vec{DWP}}}) \\
    \vec{t}_\mathrm{actual,\vec{DWP}} &= \vec{t}_\mathrm{recorded,\vec{DWP}} + t_\mathrm{delay} \\
    t_\mathrm{delay} &\sim HalfNormal(\sigma = 0.1)
\end{aligned}
$$

This well-wise rate coefficient $\vec{k}_{\vec{t},\vec{DWP}}\ [mM/h]$ depends on three factors.
The first is the concentration of the whole-cell biocatalyst $\vec{X}_{\vec{t},\vec{DWP}}\ [g_{CDW}]$ as obtained from the biomass model described above.
The second factor is the biocatalyst' specific activity $\vec{k}_\mathrm{\vec{design}}\ [mM/h/g_{CDW}]$ that depends on the experimental design of the expression phase.
The third factor is a batch-wise random effect $\vec{F}_{\vec{DASGIP}}\ [-]$ to account remaining experimental variability.

$$
\begin{aligned}
    \vec{k}_{\vec{t},\vec{DWP}} &\sim LogNormal( ln(\mu_{\vec{k}_{\vec{t},\vec{DWP}}}), 0.05)  \\
    \mu_{\vec{k}_{\vec{t},\vec{DWP}}} &= \mathrm{\vec{X}_{\vec{t},\vec{DWP}}} \cdot f_1(\vec{k}_\mathrm{\vec{design}}) \cdot f_2(\vec{F}_\mathrm{\vec{DASGIP}}) \\
    \vec{F}_\mathrm{\vec{DASGIP}} &\sim LogNormal(0, 0.1) \\
    \textrm{with} \\
    f_1 &: \vec{design} \rightarrow \vec{DWP} \\
    f_1 &: \vec{DASGIP} \rightarrow \vec{DWP} \\
\end{aligned}
$$

For the overall bioprocess optimization study we were interested in two quantities:
Design-wise specific activity $\vec{k}_{\vec{design}}$ and an experiment-independent absolute initial reaction rate $\vec{v}_{0,\vec{design}}$.
The $\vec{k}_{\vec{design}}$ parameter is part of the above equation and modeled by a two-dimensional Gaussian process to allow for inter- and extrapolation.
$\vec{k}_{\vec{design}}$ is strictly positive and we expect it around $0.1-0.8\ [mM/h]$.
The model described below achieves both properties by describing a GP for $ln(\vec{k}_{\vec{design}})$ and assigning a corresponding prior for the kernel variance $\sigma$.
The prior for lenghtscales $\vec{\ell}$ was centered on the 1/2 of the $log_{10}$ range (upper minus lower bound) of the design space.
A similar structure was used earlier for the $\mathrm{\vec{X}_{factor,\vec{glc}}}$ variable in the upstream biomass model.

$$
\begin{aligned}
    ln(\vec{k}_{\vec{design}}) &= f_{\vec{logk}_{\vec{design}}}(log_{10}(\vec{D}_{\vec{design}})) \\
    f_{\vec{logk}_{\vec{design}}}(d) &\sim GP(0, k(d,d')) \\
    k(d,d') &= \sigma^2 \cdot e^{-\frac{(d-d')^2}{2\vec{\ell}^2}} \\
    \sigma &\sim LogNormal(ln(0.7), 0.2) \\
    \vec{\ell} &\sim LogNormal(ln(\mathrm{\vec{range}}), 0.1) \\
    \vec{\mathrm{range}} &= (0.681, 2.125)^T
\end{aligned}
$$

Finally, the initial reation rate metric $\vec{v}_{0,\vec{design}}\ [mM/h]$ is derived from model parameters that do not depend on batch/reactor/replicate-specific variables:

$$
\begin{aligned}
    \vec{v}_{0,\vec{design}} &= \vec{k}_{\vec{design}} \cdot \mathrm{\vec{X}_{end,\vec{design}}} \\
    \mathrm{\vec{X}_{end,\vec{design}}} &= \mathrm{X_{end,batch}} \cdot \mathrm{\vec{X}_{factor,\vec{glc}}}
\end{aligned}
$$

