(TeX-add-style-hook
 "taxation"
 (lambda ()
   (TeX-run-style-hooks
    "../Tables/taxresultsPY"
    "../Tables/taxresultsLC")
   (LaTeX-add-labels
    "sec:Tax"))
 :latex)

