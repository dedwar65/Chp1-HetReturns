(TeX-add-style-hook
 "Chp1-presentation"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("biblatex" "backend=bibtex" "style=authoryear")))
   (TeX-run-style-hooks
    "latex2e"
    "Tables/calibPY"
    "Tables/calibLC"
    "beamer"
    "beamer10"
    "inputenc"
    "biblatex"
    "dirtytalk"
    "cancel"
    "graphicx"
    "wrapfig"
    "caption"
    "booktabs"
    "adjustbox"
    "amssymb")
   (LaTeX-add-bibliographies
    "Chp1"))
 :latex)

