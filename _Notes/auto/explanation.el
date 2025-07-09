(TeX-add-style-hook
 "explanation"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "footnote"))
 :latex)

