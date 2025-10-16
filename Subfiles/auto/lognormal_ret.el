(TeX-add-style-hook
 "lognormal_ret"
 (lambda ()
   (TeX-run-style-hooks
    "Tables/Lognorm_PY_LC_wealth_distribution_compare"
    "Tables/lognorm_tax_welfare"
    "Tables/Lognorm_PY_per_type_welfare"
    "Tables/Lognorm_LC_per_type_welfare")
   (LaTeX-add-labels
    "sec:lognorm"
    "fig:PYLognorm"
    "fig:LCLognorm"
    "fig:LognormPYLCMPCCompare"
    "fig:SimLorenzTarPoint"
    "fig:SimLorenzTarDist"))
 :latex)

