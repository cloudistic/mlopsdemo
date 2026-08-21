Installation:
	pip install dvc

Initialize DVC.
	git init
	dvc init ( git init )

Get Data and commit the changes. Also follow the onscreen instructions after every command.
    dvc get https://github.com/iterative/dataset-registry get-started/data.xml -o data/data.xml
    dvc add data/data.xml
    git add data/data.xml.dvc data/.gitignore

    git commit
    mkdir ~/projects/lectures/dvc-storage
    dvc remote add -d tutremote ~/projects/lectures/dvc-storage
    dvc push

Delete the files and try to pull from DVC.
	    rm -rf .dvc/cache
        rm -f data/data.xml
        dvc pull

Install DVC Extension in Vscode ( Optional )

Build Pipelines:
	
	Download the code and extract it.

		wget https://code.dvc.org/get-started/code.zip
		unzip code.zip && rm -f code.zip
	
	Add the following stages to the pipeline.

		dvc stage add -n prepare \
					-p prepare.seed,prepare.split \
					-d src/prepare.py -d data/data.xml \
					-o data/prepared \
					python src/prepare.py data/data.xml

		dvc stage add -n featurize \
					-p featurize.max_features,featurize.ngrams \
					-d src/featurization.py -d data/prepared \
					-o data/features \
					python src/featurization.py data/prepared data/features
		dvc stage add -n train \
					-p train.seed,train.n_est,train.min_split \
					-d src/train.py -d data/features \
					-o model.pkl \
					python src/train.py data/features model.pkl
		git add .gitignore data/.gitignore dvc.yaml
		git commit -m "pipeline defined"   
	
	Visualize the pipeline
		dvc dag
	
	Run/Reproduce the pipeline
		dvc repro
		
	Add and track experiments.
		dvc stage add -n evaluate \
		-d src/evaluate.py -d model.pkl -d data/features \
		-o eval \
		python src/evaluate.py model.pkl data/features

	Run/Reproduce the pipeline
		dvc repro

	git add .gitignore dvc.yaml dvc.lock eval
    git commit -a -m "Create evaluation stage"

    dvc metrics show
	
	Review the metrics on the DVC extensions (Optional).