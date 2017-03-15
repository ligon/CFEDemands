.PHONY
tangle: Empirics/cfe_estimation.org
	(cd Empirics; ../tangle.sh cfe_estimation.org)

.PHONY
dist: setup.py tangle
	python setup.py sdist bdist_wininst

.PHONY
upload: dist
	python setup.py upload
