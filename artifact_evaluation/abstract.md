This is the repeatability evaluation package for the tool paper "Verse: A Python library for reasoning about multi-agent hybrid system scenarios". 

The artifact is a virtual machine that contains instruction and software to reproduce all experiment results in the paper.

The admin password of the virtual machine is 
```
cav2023-re
```
The <code>README.txt</code> file in the artifact zip file contains instruction to reproduce all experiments in the paper. A PDF version of the file can be found in <code>artifact_evaluation/artifact_evaluation.pdf</code>.

# Artifact URLs

The DOI of the artifact is 
```
10.6084/m9.figshare.22679485
```

The link to the artifact on Figshare is:
    https://figshare.com/articles/software/Verse_A_Python_library_for_reasoning_about_multi-agent_hybrid_system_scenarios/22679485

The link to the artifact on Google drive is: 
    https://drive.google.com/file/d/1SfABQ1bkFXijCpANfODQAMdvFnBXpw0a/view?usp=sharing

A detailed interactive tutorial for verse can be found at:
    https://github.com/AutoVerse-ai/Verse-library/blob/tutorial/tutorial/Verse_Tutorial_Drone.ipynb
and a PDF version of the tutorial is included in the artifact as "tutorial.pdf".

The tool is publicly available at
    https://github.com/AutoVerse-ai/Verse-library

# Artifact SHA

The SHA of the artifact zip file is 
```
f480732cbda706f9dffc967005813af0ebcef48e50f1417a5b50871d9cb14ceb
```

# Artifact Reusability

The software is available outside of the provided virtual machine. It can be installed on any machine (Windows, MAC, Linux) with Python3.8+.

To install the library, first clone the repository from github 
```
https://github.com/AutoVerse-ai/Verse-library.git
```
Then go to the root directory of the artifact 
```
cd Verse-library
```
and install the library using pip 
```
python3 -m pip install -e .
```
There are many examples in the <code>./demo</code> folder. To run an example, one can in the root directory of library run
```
python3 demo/cav2023/exp1/exp1.py
```

## Tutorial
The library also comes with an interactive tutorial for how to run create scenarios using the library. The tutorial is located at <code>./tutorial/tutorial.ipynb</code>. To run the tutorial, first go to the tutorial folder 
```
cd tutorial
```
Then install additional requirements for the tutorial using command 
```
python3 -m pip install -r requirements_tutorial.txt
```
The tutorial can then be run using jupyter notebook

A pdf version of the same tutorial is also provided in <code>./tutorial/tutorial.pdf</code>