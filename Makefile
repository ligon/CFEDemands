.PHONY: tangle sdist wininst wheel upload localinstall clean CHANGES.txt

tangle: Empirics/cfe_estimation.org
	(cd Empirics; ../tangle.sh cfe_estimation.org)
	(cd Demands; ../tangle.sh demands.org)
	(cd Demands; ../tangle.sh monotone_function_solver.org)

sdist: setup.py tangle CHANGES.txt
	python setup.py sdist 

wininst: setup.py tangle CHANGES.txt
	python setup.py bdist_wininst

wheel: setup.py tangle CHANGES.txt cfe/requirements.txt
	pip wheel --wheel-dir=dist/ .

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

cfe/requirements.txt:
	(cd cfe; pigar)

localinstall: clean wheel
	(cd dist; pip install CFEDemands*.whl --upgrade)

upload: wheel
	#python setup.py sdist upload
	twine upload dist/*

clean: 
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
