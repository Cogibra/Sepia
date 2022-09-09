# Sepia

<div align="center">
<img src="https://github.com/cogibra/sepia/raw/master/assets/sepia_sketch.jpg" width=50%>
</div>

Neuro/symbolic models for learning and reasoning about biological sequences and structures.

## Getting started

### Setup

**Clone**

```
git clone git@github.com:Cogibra/Sepia.git 
```

**Install**

```
cd Sepia
virtualenv my_env --python=python3.8
source my_env/bin/activate
pip install -e .
```

**Test**

```
python -m sepia.test.test_all
#OR
coverage run -m sepia.test.test_all && coverage report
```

