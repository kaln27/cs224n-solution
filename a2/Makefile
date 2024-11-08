PAPER=main

all:
	cd latex && pdflatex -shell-escape $(PAPER)
	cd latex && pdflatex -shell-escape $(PAPER)
	
clean:
	find $(dirname "$0/latex") -regextype posix-extended  -regex ".*\.(aux|log|out|thm|toc|bbl|blg|fdb_latexmk|fls|gz|snm|nav|vrb)" -delete
	rm latex/_* -rf

.PHONY: clean