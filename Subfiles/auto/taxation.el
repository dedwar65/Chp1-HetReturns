(TeX-add-style-hook
 "taxation"
 (lambda ()
   (TeX-run-style-hooks
    "../Tables/taxresultsPY"
    "../Tables/taxresultsLC"
    "Tables/unif_tax_welfare"
    "Tables/unif_PY_per_type_welfare"
    "Tables/unif_LC_per_type_welfare")
   (LaTeX-add-labels
    "sec:Tax"))
 :latex)

