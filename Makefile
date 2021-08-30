ORG_INPUTS = Empirics/cfe_estimation.org Empirics/result.org \
             Demands/demands.org Demands/engel_curves.org    \
             Demands/monotone_function_solver.org\
	     Files/input_files.org

.PHONY: tangle wheel upload localinstall devinstall clean test CHANGES.txt

all: tangle test devinstall wheel

tangle: .tangle

.tangle: $(ORG_INPUTS) 
	(cd Empirics; ../tangle.sh cfe_estimation.org)
	(cd Empirics; ../tangle.sh result.org)
	(cd Files; ../tangle.sh input_files.org)
	(cd Demands; ../tangle.sh demands.org)
	(cd Demands; ../tangle.sh engel_curves.org)
	(cd Demands; ../tangle.sh monotone_function_solver.org)
	touch .tangle

test: .test 

.test: $(ORG_INPUTS)
	pytest cfe/test/
	touch .test

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
	-rm -f .test
	-rm -f .tangle
