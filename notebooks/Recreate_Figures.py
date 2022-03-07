import pipeline
import pathlib

wd = pathlib.Path(r"C:\Git Repos\CAR_Paper\diginbio-car-paper\results")
#pipeline.plot_ph(wd)
pipeline.plot_btm_overview(wd)
pipeline.plot_experiment_design(wd)
