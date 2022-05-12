# devneuronsMorph

Explain what this codebase is for; goals, usage, papers, resources etc., here.
This README should answer any questions you think you might get bored of
being asked by other users, and anything you have ever struggled to remember
(e.g. how to set it up).

This repository was based on a template at https://github.com/navis-org/pymaid_template .

## Template usage

This section and below can be deleted once read.
Modify the above but please keep the attribution line.

### First use

```sh
# Pick a name for your project
PROJECT_NAME="my_project"

# Clone this template, then change directory into it
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

### General use

- Whenever you have a new terminal, activate the environment: `source venv/bin/activate`
- Run the script with `python script.py`

### Structure

This template is designed to minimise clutter in the top level directory, keeping it legible.

The entry point here is `./script.py`, but you can add more scripts for different functionality.
Shared functionality should be added to module files in the `utils/` package.
This is also a good way to keep your entry point script legible at a high level.

Sometimes you download a lot of data from live CATMAID instances,
which can be slow and lead to different results on subsequent runs.
You can cache that data in the `cache/` directory: git will ignore these files.

CATMAID (and other) credentials must never enter your version control system.
Git will ignore anything in the `credentials/` directory (except `example.json`):
load your credentials from files there.

Output should be stored in the `output/` directory to decrease clutter.
For now, contents of this directory is git-ignored too,
although sometimes you may want to track output: create a new directory or edit `output/.gitignore`.

Document what your project is for in this `README.md`.

Python dependencies are in `requirements.txt`:
keep this up to date as you use more packages so that it's easy to recreate your environment.

`.github/workflows/` contains configuration to have GitHub automatically check your code
for some formatting/ legibility issues and some types of bug automatically when you push.

### General guidelines

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

### License

This template is licensed under the Unlicense, but your code should not be;
that's why it's deleted.
If your code is ever released, or should be used by someone you haven't explicitly given permission to,
you should provide your own license.
