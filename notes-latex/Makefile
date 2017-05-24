NOTES = cs112.pdf cs118.pdf classics30.pdf

all: $(NOTES)
	rm -rf *.aux *.log

%.pdf: %.tex
	pdflatex $^

clean:
	rm -rf *.aux *.log *.pdf
