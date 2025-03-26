# Lightning template

How to structure the code of a deep learning project using pytorch lightning

## How to start
- In GitLab:
  - [Fork][fork] this repo into YOUR personal workspace! <img src="fork_new.png" height="50">
  - Change the name of the project: "Settings" -> "General project" -> "Project name" : `YOUR_PROJECT`
  - Change the url of the project: "Settings" -> "Advanced" -> "Rename repository" -> "Path" : `YOUR_PROJECT` (same as above)
  - Remove fork relationship: "Settings" -> "Advanced" -> "Remove fork relationship"


- In your computer:
  - If all of the above was done, you can now happily clone `YOUR_PROJECT`
  - Run `python3 install` in the root directory to install the package. You will be asked for how you want to name your python package.

## It's just a package
With the `install.py` we have installed the package which allows us to import it from anywhere in the system like every other package: 

``` py
    import YOUR_PROJECT.models.models
```

You don't have to reinstall anything after changing your code.


## Directory structure

``` bash
└── lightning_project
    ├── project_name
    │   ├── experiments
    │   ├── config
    │   │   └── config.yaml
    │   ├── datasets
    │   │   ├── datasets.py
    │   │   └── __init__.py
    │   ├── models
    │   │   ├── blocks.py
    │   │   ├── __init__.py
    │   │   ├── loss.py
    │   │   └── models.py
    │   ├── test.py
    │   ├── train.py
    │   └── utils
    │       └── __init__.py
    ├── README.md
    ├── requirements.txt
    ├── install.py
    └── setup.py
```

