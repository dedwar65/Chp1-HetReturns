(TeX-add-style-hook
 "Chp1-draft"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "Subfiles/packages"
    "Subfiles/abstract"
    "Subfiles/intro"
    "Subfiles/litrev"
    "Subfiles/model"
    "Subfiles/results"
    "article"
    "art10"
    "subfiles"
    "footnote"
    "lipsum"))
 :latex)

