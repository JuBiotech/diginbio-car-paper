import pathlib
import pipeline


def run_all(workdir):
    pipeline.load_layout(workdir)
    pipeline.plot_experiment_design(workdir)
    pipeline.plot_btm_overview(workdir)
    pipeline.plot_ph(workdir)
    pipeline.plot_o2(workdir)

    pipeline.load_biomass_calibration(workdir)

    pipeline.fit_biomass_calibration(workdir, wavelength=360)
    pipeline.fit_biomass_calibration(workdir, wavelength=600)
    pipeline.fit_product_calibration(workdir)
    pipeline.plot_biomass_calibration(workdir, wavelength=360)
    pipeline.plot_biomass_calibration(workdir, wavelength=600)
    pipeline.plot_product_calibration(workdir)

    pipeline.load_observations(workdir)

    pipeline.plot_A600_kinetics(workdir)

    pipeline.fit_model(workdir)
    pipeline.compute_diagnostics(workdir)
    pipeline.check_convergence(workdir)

    pipeline.plot_trace(workdir)
    pipeline.plot_run_effect(workdir)
    for var_name in [
        "ls_X",
        "scaling_X",
        "Xend_batch",
        "Xend_dasgip",
        "S0",
        "time_delay",
        "ls_s_design",
        "scaling_s_design",
        "run_effect",
        "s_design_mean",
    ]:
        pipeline.plot_prior_vs_posterior(workdir, var_name=var_name)

    pipeline.plot_3d_by_design(workdir, var_name="s_design")
    pipeline.plot_3d_by_design(workdir, var_name="k_design")

    pipeline.sample_gp_X_factor_pp(workdir)
    pipeline.plot_gp_X_factor_pp(workdir)

    pipeline.sample_gp_metric_posterior_predictive(workdir)
    for metric in ["s_design", "k_design"]:
        pipeline.plot_gp_metric_posterior_predictive(workdir, var_name=f"dense_{metric}")
        pipeline.plot_gp_metric_pp_interval(workdir, var_name=f"dense_{metric}")

    pipeline.sample_gp_metric_pp_crossection(workdir)
    for metric in ["s_design", "k_design"]:
        pipeline.plot_gp_metric_crossection(workdir, var_name=f"dense_{metric}")

    pipeline.export_summary(workdir)
    pipeline.plot_kinetics(workdir)
    pipeline.plot_p_best_dual_heatmap(workdir)
    pipeline.plot_p_best_single_heatmap(workdir, metric="s_design")
    best_k_design = pipeline.plot_p_best_single_heatmap(workdir, metric="k_design")
    pipeline.predict_units(workdir, design_log10=best_k_design)
    pipeline.sample_pp_best_tested_vs_predicted(workdir)
    pipeline.report_best_tested_vs_predicted(workdir)
    pipeline.plot_p_best_tested(workdir)
    return


if __name__ == "__main__":
    workdir = pathlib.Path(__file__).absolute().parent.parent / "results"
    workdir.mkdir(exist_ok=True)
    run_all(workdir)
