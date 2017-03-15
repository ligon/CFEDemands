.PHONY: tangle sdist wininst upload
tangle: Empirics/cfe_estimation.org
	(cd Empirics; ../tangle.sh cfe_estimation.org)

sdist: setup.py tangle
	python setup.py sdist

wininst: setup.py tangle
	python setup.py bdist_wininst

upload: sdist wininst
	python setup.py upload
