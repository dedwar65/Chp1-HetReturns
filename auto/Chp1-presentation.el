(TeX-add-style-hook
 "Chp1-presentation"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("biblatex" "backend=bibtex" "style=authoryear")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
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

