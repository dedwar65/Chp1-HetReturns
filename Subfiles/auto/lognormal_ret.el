(TeX-add-style-hook
 "lognormal_ret"
 (lambda ()
   (TeX-run-style-hooks
    "Tables/Lognorm_PY_LC_wealth_distribution_compare"
    "../Tables/lognormtaxresultsPY"
    "../Tables/lognormtaxresultsLC"
    "Tables/lognorm_tax_welfare"
    "Tables/Lognorm_PY_per_type_welfare"
    "Tables/Lognorm_LC_per_type_welfare"
    "Tables/lognorm_elasticities")
   (LaTeX-add-labels
    "sec:lognorm"
    "fig:PYLCLognorm"
    "fig:LognormPYLCMPCCompare"
    "fig:EmpLorenzTar"
    "fig:SimLorenzTarDist"))
 :latex)

