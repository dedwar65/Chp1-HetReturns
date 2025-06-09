(TeX-add-style-hook
 "results"
 (lambda ()
   (TeX-run-style-hooks
    "../Tables/calibPY"
    "../Tables/calibLC")
   (LaTeX-add-labels
    "sec:results"
    "fig:PYUnif"
    "fig:LCUnif"
    "fig:EmpLorenzTar"
    "fig:SimLorenzTarPoint"
    "fi