# devneuronsMorph


This repository contains scripts and files for a systematic quantification of morphological similarity across homologous (left:right) pairs of neurons. As a result, any 'deviant' (hence, dev) neurons - which significantly differ from expected morphological alikeness - can be identified from low scores. 

Utilises data stored and generated within CATMAID, interfaced with via the Python libraries NAVis and pymaid. There are two separate scripts to quantify morphological similarity: one for NBLAST (https://navis.readthedocs.io/en/latest/source/tutorials/nblast.html) and one for SynBLAST (https://navis.readthedocs.io/en/latest/source/generated/navis.synblast.html?highlight=synblast). Files all utilise connectomic data from 'Seymour', an L1 D. melangoaster larvae.

Links: CATMAID (https://catmaid.readthedocs.io/en/stable/) ; NAVis (https://navis.readthedocs.io/en/latest/) ; pymaid (https://pymaid.readthedocs.io/en/latest/)

Also see devneuronsConn repo


## Files and scripts


cns-pairs.csv - list of 1640 pairs (left and right) of homologous neurons, represented as CATMAID skeleton IDs (skids). Some are annotated with CNS region. 
brain_landmarks.csv - landmarks from left and right hemispheres of brain; facilitates transformation of one side of homologous pair member onto other
CNS_landmark.csv - identical role to above, but includes both brain and VNC landmarks for whole CNS transformation
requirements.txt - dependencies required to run scripts

script_nblast.py - takes pairs and calculates a cross-validated NBLAST similarity score for each. Different options for Strahler pruning and node resampling, which will be represented in output file. Outputs CSV with left neuron skid, right neuron skid, partition, score and left neuron name (usually very similar to right homologue) 
script_synblast.py - very similar approach to above, but instead utilises SynBLAST to quantify morphological similarity. No Strahler pruning or node resampling as a result, and any neurons with no synapses will have their pair filtered out.
explore_morph.py - script to perform exploratory analysis on data; generates various plots and can be modified for bespoke analyses

cache and output folders to locally store respective components after running script.


## General information


CATMAID requires credentials, which must be provided by user from a locally stored directory. These will not be tracked by repository's git.

Live CATMAID instances can require lots of data to be downloaded, which both slows scripts and can lead to different results on subsequent runs (if neurons are modified). Intermediate data downloaded from CATMAID locally will be stored within `cache/` directory; git will ignore these files.

Output CSVs and exploratory analyses will be stored in the `output/` directory to decrease clutter. This is also currently git ignored, but you can modify any of these if desired.


TO DO: modify below


## First use


```sh
# Pick a name for your project
PROJECT_NAME="my_project"

# Clone this , then change directory into it
git clone https://github.com/navis-org/pymaid_template.git "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Delete the template's git history and license
rm -rf .git/ LICENSE
# Initialise a new git repo
git init
# Commit the existing files so you can track your changes
git add .
git commit -m "Template from navis-org/pymaid_template"

# Ensuring that you are using a modern version of python (3.9, here), create and activate a virtual environment
python3.9 -m venv --prompt "$PROJECT_NAME" venv
source venv/bin/activate
# use `deactivate` to deactivate the environment

# Install the dependencies
pip install -r requirements.txt

# Make your own credentials file
cp credentials/example.json credentials/credentials.json
```

Then edit the credentials file to access your desired CATMAID server, instance, and user.


## General use


- Whenever you have a new terminal, activate the environment: `source venv/bin/activate`
- Run the script with `python script.py`


## General guidelines


- Use a modern python. Many actively maintained scientific tools follow numpy's deprecation schedule: https://numpy.org/neps/nep-0029-deprecation_policy.html
- Follow coding standards to make your code as legible and recogniseable as possible. See PEP8: https://www.python.org/dev/peps/pep-0008/
  - Coding standards sound like nitpicking but they really, really help. e.g. "I know I wrote a function to get data, but was it called getData, GetData, get_data, GET_DATA, or what?". If code is PEP8-compliant, there is only one answer
  - Auto-formatters (e.g. `black`, `isort`) are great for legibility and consistency: use `make format` to format this repository.
  - Linters (e.g. `flake8`) can detect a number of bugs, possible coding issues, and questionable formatting: use `make lint` to lint this repository (format first).
- Documentation makes your code much easier to understand for the next person to read it: that person will probably be you, so it's worthwhile.
  - Type hints, especially on functions, are also great: https://realpython.com/python-type-checking/#hello-types
  - Docstrings at the top of modules and functions are better than comments, as they are accessible by the `help()` function in a python REPL
- Use seeded random number generators where randomness is needed, for replicability.
- Remember to use ISO-8601 dates/times (YYYY-MM-DD) where necessary.


## License


This repository was based on a template at https://github.com/navis-org/pymaid_template .

