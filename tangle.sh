#!/bin/sh
# -*- mode: shell-script -*-
#
# tangle files with org-mode
#
DIR=`pwd`
FILES=""

# wrap each argument in the code required to call tangle on it
for i in $@; do
    FILES="$FILES \"$i\""
done

emacs -Q --batch \
     --eval "(progn
     (require 'org)(require 'ob)(require 'ob-tangle)
     (setq org-src-preserve-indentation t)
     (mapc (lambda (file)
            (find-file (expand-file-name file \"$DIR\"))
            (org-babel-tangle)
            (kill-buffer)) '($FILES)))" 2>&1 |grep -i tangled
