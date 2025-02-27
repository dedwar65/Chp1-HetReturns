(TeX-add-style-hook
 "packages"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("biblatex" "backend=bibtex" "style=authoryear") ("fontenc" "T1") ("caption" "font=small" "format=plain" "labelfont=bf" "textfont=normal" "justification=justified" "singlelinecheck=false") ("nth" "super")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "footmisc"
    "datetime2"
    "refcount"
    "xr-hyper"
    "hhline"
    "rotating"
    "makecell"
    "subfiles"
    "inputenc"
    "biblatex"
    "dirtytalk"
    "cancel"
    "graphicx"
    "wrapfig"
    "caption"
    "booktabs"
    "adjustbox"
    "fontenc"
    "sectsty"
    "amsmath"
    "amssymb"
    "amsfonts"
    "placeins"
    "nth"
    "dcolumn"
    "subcaption"
    "color"
    "afterpage")
   (TeX-add-symbols
    '("mpurple" 1)
    '("mblue" 1)
    '("mred" 1)
    '("myblue" 1)
    '("myred" 1)
    '("notinsubfile" 1)
    '("onlyinsubfile" 1)
    "onlyinsubfile"
    "notinsubfile"
    "ifCDC"
    "exm"
    "exmm"
    "obs"
    "obss"
    "cor"
    "corr"
    "ex"
    "exx"
    "brmk"
    "ermk"
    "thm"
    "nt"
    "thmm"
    "lm"
    "lmm"
    "ass"
    "asss"
    "df"
    "dff"
    "prp"
    "prpp"
    "bqu"
    "equ"
    "eq"
    "eqq"
    "cl"
    "cll"
    "bit"
    "eit"
    "ben"
    "een"
    "bcen"
    "ecen"
    "fn"
    "ds"
    "dss"
    "prf"
    "prff"
    "cs"
    "css"
    "ml"
    "lb"
    "ra"
    "tra"
    "supp"
    "inff"
    "nf"
    "mmax"
    "mmin"
    "uhr"
    "CR"
    "CC"
    "CT"
    "CS"
    "CM"
    "CL"
    "CP"
    "CN"
    "red"
    "green"
    "blue"
    "purple"
    "medcap"
    "medcup")
   (LaTeX-add-environments
    '("customdf" 1)
    '("customlm" 1)
    '("ecustomcor" 1)
    '("customcor" 1)
    '("ecustomthm" 1)
    '("customthm" 1)
    "tm"
    "dfn"
    "lma"
    "assu"
    "prop"
    "cro"
    "example"
    "observation"
    "exa"
    "remak"
    "ax"
    "claim"
    "innercustomthm"
    "einnercustomthm"
    "innercustomcor"
    "einnercustomcor"
    "innercustomlm"
    "innercustomdf")
   (LaTeX-add-bibliographies
    "Chp1")
   (LaTeX-add-array-newcolumntypes
    "d"))
 :latex)

