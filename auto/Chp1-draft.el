(TeX-add-style-hook
 "Chp1-draft"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "titlepage")))
   (TeX-run-style-hooks
    "latex2e"
    "Subfiles/packages"
    "article"
    "art10"
    "subfiles"
    "athblk"
    "footnote"
    "lipsum"))
 :latex)

