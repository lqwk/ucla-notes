NOTES = cs112.pdf cs118.pdf

all: $(NOTES)
	rm -rf *.aux *.log

%.pdf: %.tex
	pdflatex $^

clean:
	rm -rf *.aux *.log *.pdf
