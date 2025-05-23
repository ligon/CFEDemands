ORG_INPUTS = Empirics/cfe_estimation.org \
	     Empirics/regression.org Files/input_files.org

.PHONY: tangle wheel upload localinstall devinstall clean test CHANGES.txt

all: tangle test devinstall wheel

tangle: .tangle

.tangle: $(ORG_INPUTS) 
	(cd Empirics; ../tangle.sh cfe_estimation.org)
	#(cd Empirics; ../tangle.sh result.org)
	(cd Empirics; ../tangle.sh regression.org)
	(cd Files; ../tangle.sh input_files.org)
	touch .tangle

test: .test 

.test: .tangle
	pytest cfe/test/
	touch .test

wheel: pyproject.toml tangle test CHANGES.txt #cfe/requirements.txt
	pip wheel --wheel-dir=dist/ .

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

cfe/requirements.txt:
	(cd cfe; pigar generate)

localinstall: clean wheel
	(cd dist; pip install CFEDemands*.whl) # --upgrade)

devinstall: tangle test 
	pip install -e .

upload: wheel
	twine upload dist/cfedemands*.whl

clean: 
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f cfe/requirements.txt
	-rm -f CHANGES.txt
	-rm -f .test
	-rm -f .tangle
	-rm -f cfe/test/*.py
	-rm -f cfe/stochastic_test/*.py
