.PHONY: png pdf test test_fast lint figures clean
SOURCES := $(wildcard figs/*.dot)
PNGS = $(SOURCES:.dot=.png)
PDFS = $(SOURCES:.dot=.pdf)

%.png: %.dot
	dot -Tpng $*.dot -o $@

%.pdf: %.dot
	dot -Tpng $*.dot -o $@

all: png pdf

png: $(PNGS)

pdf: $(PDFS)

data: data.zip
	unzip -u data.zip

test: data
	python -m pytest

test_fast: data
	python -m pytest -k cp

lint:
	flake8 agcounts tests
	mypy agcounts tests
	pydocstyle agcounts

figures: data
	python generate_plots.py

clean:
	rm -Rf data
	rm -Rf figs/*.pdf figs/*.png

