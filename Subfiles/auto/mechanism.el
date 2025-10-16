(TeX-add-style-hook
 "mechanism"
 (lambda ()
   (TeX-run-style-hooks
    "Tables/unif_elasticities")
   (LaTeX-add-labels
    "sec:Mechanism"))
 :latex)

