ORG_INPUTS = Empirics/cfe_estimation.org \
	     Empirics/regression.org Files/input_files.org

.PHONY: tangle wheel build upload localinstall devinstall clean test CHANGES.txt release

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
	poetry run pytest cfe/test/
	touch .test

build: pyproject.toml tangle test CHANGES.txt
	rm -rf dist
	poetry build

wheel: build

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

localinstall: clean wheel
	poetry install

devinstall: tangle test
	poetry install

upload: wheel
	poetry publish

# Usage: make release BUMP=patch  (or minor, major, prepatch, etc.)
# Bumps and tags only; it deliberately does not build, because the artifact must
# come from the bumped version.  Publishing to PyPI is done by the
# publish-to-pypi workflow when a GitHub Release is published; 'make upload' is
# the manual fallback.
BUMP ?= patch
release: tangle test
	$(eval NEW_VER := $(shell poetry version $(BUMP) -s))
	git add pyproject.toml
	git commit -m "Bump version to $(NEW_VER)"
	git tag v$(NEW_VER)
	@echo "Tagged v$(NEW_VER). Run 'git push && git push --tags', then publish a"
	@echo "GitHub Release for v$(NEW_VER) to build and upload to PyPI."

clean:
	-rm -f dist/*.tar.gz dist/*.exe dist/*.whl
	-rm -f CHANGES.txt
	-rm -f .test
	-rm -f .tangle
	-rm -f cfe/test/*.py
	-rm -f cfe/stochastic_test/*.py
