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
Reference wells of known product concentrations, but without 2-amino benzoic acid are also included in the model, albeit with the assumption that the 2-amino benzaldehyde concentration remains constant over time.

A likelihood needed for parameter inference by Markov-chain Monte Carlo (MCMC) is created from process model predictions and observed data according to the relationships described by the separately fitted calibration models.
At 600&nbsp;nm this is simply the likelihood of the observed data given the predicted biomass concentration.
At 360&nbsp;nm however, both biomass and ABAO reaction product absorb and therefore the sum of their absorbances needs to be taken into account for the likelihood.

After this high-level description, the next chapter introduces the most relevant components and model variables.

### Biomass process model
The biomass in the experiment is sourced from a "seed train" of cultivations in three different scales and operating modes:
1. 1&nbsp;L DASGIP batch bioreactor; 1 per experiment run.
2. 10&nbsp;mL 2mag fed-batch macrobioreactor; 48 per experiment run.
3. 600&nbsp;µL biotransformation in square deep-well plate; 60 per experiment run.

The process model must describe biomass in each biotransformation well so it can be accounted for in the 360&nbsp;nm absorbance.
Since a universally activity metric, that can be interpredeted independent from experimenteal batch effects is desired, the model must additionally describe biomass in a way that excludes random batch effects.
The first process stage at which such an experiment-independent prediction is needed, is the final biomass concentration of the 1&nbsp;L batch cultivation.

Concretely, we describe the per-experiment final biomass concentration at the 1&nbsp;L scale as a LogNormal-distributed variable called $\mathrm{\vec{X}_{end,DASGIP}}$ with an entry for each experiment run.
To obtain an experiment-independent prediction, we introduced $\mathrm{X_{end,batch}}$ as a _group mean prior_, also known as a _hyperprior_, around which the $\mathrm{\vec{X}_{end,DASGIP}}$ is centered.
The prior on $\mathrm{X_{end,batch}}$ is weakly (large $\sigma$) centered at $0.5\ g/L$, whereas actual batches should only deviate from that group mean by ca. $5\ \%$.

$$\begin{aligned}
    \mathrm{X_{end,batch}} &\sim LogNormal(\mu=ln(0.5), \sigma=0.5)\\
    \mathrm{\vec{X}_{end,DASGIP}} &\sim LogNormal(\mu=ln(\mathrm{X_{end,batch}}), \sigma=0.05)
\end{aligned}$$

This hierarchical structure is a common motif in Bayesian modeling since it enables a model to learn variables that are essential to the process understanding (here: $\mathrm{X_{end,batch}}$) while retaining the ability to describe the fine-grained structure of the experimental data (here: $\mathrm{\vec{X}_{end,DASGIP}}$).
The motif of hierarchically modeled variables was used in several places of our bioprocess model.
For a thorough introduction to hierarchical modeling, we recommend [betancourt2020].

The second process stage in the biomass seed train is the expression in 10&nbsp;mL scale under fed-batch conditions.
Every 10&nbsp;mL 2mag reactor was inoculated with culture broth from a DASGIP reactor, hence a mapping $f_\mathrm{run \rightarrow reactor}$ yields initial biomass concentrations $\mathrm{X_{start,2mag}}$ by sub-indexing the $\mathrm{\vec{X}_{end,DASGIP}}$ variable.
The experimental design of the fed-batches comprised varying glucose feed rates and IPTG concentrations.
It is plausible to assume a dependence of the final biomass concentration $\mathrm{\vec{X}_{end,2mag}}$ on the glucose feed rate.
Without any mechanistic assumptions, we lump the final biomass concentration per 1&nbsp;mL reactor as the product of initial biomass concentration with a positive factor $\mathrm{X_{factor,glc}}$ that depends on the glucose feed rate.
Dependence of $\mathrm{X_{factor,glc}}$ on the glucose feed rate is modeled by a Gaussian process such that our model can also interpolate and make predictions for new glucose feed rate settings.

$$
\begin{aligned}
    \mathrm{\vec{X}_{start,2mag}} &= f_\mathrm{DASGIP \rightarrow 2mag}(\mathrm{\vec{X}_{end,DASGIP}}) \\
    \mathrm{\vec{X}_{end,2mag}} &= \mathrm{\vec{X}_{start,2mag}} \cdot f_{\mathrm{glc-design \rightarrow 2mag}}(\mathrm{\vec{X}_{factor,glc}}) \\
    \textrm{with} \\
    ln(\mathrm{\vec{X}_{factor,glc}}) &= f_\mathrm{logX_{factor,glc}}(log_{10}(\vec{D}_\mathrm{design,glc})) \\
    f_\mathrm{logX_{factor,glc}}(d) &\sim GP(0, k(d,d')) \\
    k(d,d') &= \sigma^2 \cdot e^{-\frac{(d-d')^2}{2\ell^2}} \\
    \sigma &\sim LogNormal(ln(0.3), 0.1) \\
    \ell &\sim LogNormal(ln(0.34), 0.1)
\end{aligned}
$$

The Gaussian process was parametrized with a mean function of $0$, thereby centering the prior for $\mathrm{X_{factor,glc}}$ around $1$.
For the covariance function we chose a scaling parameter $\sigma$ such that the prior variance for the factor is around $\plusmn30\ \%$.
The prior for $\ell$ in the exponential quadratic kernel encodes a belief that $\mathrm{X_{factor,glc}}$ varies smoothly on a length scale of around half of the (logarithmic) design space.

The 3rd and final process stage is the biotransformation.
Here, the initial biomass concentration in every DWP well $\mathrm{\vec{X}_{0,DWP}}$ equals the final biomass concentration from a corresponding 10&nbsp;mL reactor.
The biomass concentration continued to change over the course of the biotransformation, because the solution also contained glucose as a carbon source.
Inspired by the $\mu(t)$ method described in [blet] we account for this biomass growth during the biotransformation with a Gaussian random walk of the discretized growth rate $\vec{\mu}_t$.
The result are biomass concentrations for every well and measurement cycle $\mathrm{\vec{X}_{t,DWP}}$.

$$
\begin{aligned}
    \mathrm{\vec{X}_{0,DWP}} &= f_\mathrm{2mag\rightarrow DWP}(\mathrm{\vec{X}_{end,2mag}}) \\
    \mathrm{\vec{X}_{t \ge 1,DWP}} &= \mathrm{\vec{X}_{0,DWP}} \cdot e^{cumsum(\vec{\mu}_t \cdot \vec{dt})}\\
    \vec{\mu}_t &\sim GaussianRandomWalk(\sigma=0.1) \\
\end{aligned}
\\
\textrm{Note: $\vec{\mu}_t$ and $\vec{dt}$ are modeled for each biotransformation well independently.}
$$

### Biotransformation reaction process model

