.PHONY: tangle wheel upload localinstall devinstall clean test

tangle: Empirics/cfe_estimation.org
	(cd Empirics; ../tangle.sh cfe_estimation.org)
	(cd Demands; ../tangle.sh demands.org)
	(cd Demands; ../tangle.sh engel_curves.org)
	(cd Demands; ../tangle.sh monotone_function_solver.org)

test:
	pytest cfe/test/

wheel: setup.py tangle test CHANGES.txt cfe/requirements.txt
	pip wheel --wheel-dir=dist/ .

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

cfe/requirements.txt:
	(cd cfe; pigar)

localinstall: clean wheel
	(cd dist; pip install CFEDemands*.whl --upgrade)

devinstall: tangle test
	pip install -e .

upload: wheel
	twine upload dist/*

clean: 
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f cfe/requirements.txt
	-rm -f CHANGES.txt
