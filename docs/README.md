# Verse Library Documentation
This forder contains the documentation template for the Verse library. The documentation is automatically generated using [sphinx](https://www.sphinx-doc.org/en/master/). 

## Prerequsites 
The following libraries are required for compile verse's document
- sphinx
- myst-parser 
- numpydoc 
- sphinx_rtd_theme

The required packages can be installed using command 
```
python3 -m pip install -r requirements-doc.txt
```

## Compiling documents
### For Linux
For linux user, the document can be compiled using command 
```
make html
```
in the ```docs/``` folder

### For Windows
For Windows user, the document can be compiled using command 
```
./make.bat html
```
in the ```docs/``` folder

## Viewing documents
The compiled result can be found in ```docs/build/html``` folder. The root of the compiled document is stored in ```docs/build/html/index.html```

## Example architectural document
An example highlevel architectural document can be found in file ```docs/source/parser.md```

## Example docstring for class/function
An example docstring for class function can be found in file ```verse/agents/base_agent.py``` for class ```BaseAgent``` and function ```BaseAgent.__init__``` and ```BaseAgent.TC_simulate```
