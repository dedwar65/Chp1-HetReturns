(TeX-add-style-hook
 "results"
 (lambda ()
   (TeX-run-style-hooks
    "../Tables/calibPY"
    "Tables/Unif_PY_wealth_distribution_compare"
    "../Tables/calibLC"
    "Tables/Unif_LC_wealth_distribution_compare")
   (LaTeX-add-labels
    "sec:results"
    "fig:PYUnif"
    "fig:LCUnif"
    "fig:PYMPCWealthDecileCompare"
    "fig:LCMPCWealthDecileCompare"
    "fig:EmpLorenzTar"
    "fig:SimLorenzTarPoint"
    "fig:SimLorenzTarDist"))
 :latex)

