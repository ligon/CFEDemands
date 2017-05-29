.PHONY: tangle sdist wininst upload localinstall clean CHANGES.txt

tangle: Empirics/cfe_estimation.org
	(cd Empirics; ../tangle.sh cfe_estimation.org)
	(cd Demands; ../tangle.sh demands.org)

sdist: setup.py tangle CHANGES.txt
	python setup.py sdist 

wininst: setup.py tangle
	python setup.py bdist_wininst

CHANGES.txt:
	git log --pretty='medium' > CHANGES.txt

localinstall: clean sdist
	(cd dist; sudo -H pip2 install CFEDemands*.tar.gz --upgrade)

upload: sdist #wininst
	python setup.py sdist upload

clean: 
	-rm -f dist/*.tar.gz dist/*.exe
