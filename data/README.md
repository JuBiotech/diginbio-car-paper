# Data Formats
## Layout file
The following rules should be followed for the XLSX sheet or `DataFrame` of layout information.

The `replicate_id` must be set as the index of the `DataFrame`.

```
replicate_id : str
    This column is optional, but recommended.
    Every value should uniquely identify a biotransformation calibration/reaction.
    If the column is present, there must be entries in all rows.
run : str
    Unique identifier of the experiment run.
    For FZJ that'll be a 6-digit hagelkorn ID.
assay_well : str
    Position of the reaction in biotransformation DWP and measurement MTP.
    Must be given in A01 format.
reactor : str
    Identifier of the vessel that produced the biomass for the biotransformation.
group : str
    Name of a group for which a common groundtruth activity shall be modeled.
    Use this field to group replicates of the same reaction conditions.
product : float
    Known product concentration in calibration wells.
    Leave cells of reaction wells blank.
biomass : float
    If available use this column to provide known biomass concentrations.
    For example in calibration wells, or if the initial biomass concentration was normalized or measured.

In addition to the above, further columns may be added for relevant condition information such as pH, glucose concentration, ...
```

## Absorbance kinetics
TUM-style CSV files of absorbances can be read with `dataloading.read_absorbances`.

The `dataloading.vectorize_observations` function concatenates dataframes of this format into big dataframes for time and absorbances.
