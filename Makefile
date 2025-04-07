data:
	python -B src/Extract_Arxiv_Sources.py

run:
	python -B src/solution.py

clean:
	rm -rf Arxiv_Resources.csv