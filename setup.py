from setuptools import setup, find_packages

#  packages=["sepia", "testing"], \

setup(\
        name="sepia", \
        version = "0.00000", \
        install_requires =[\
                "coverage==6.4.3",\
                "jax>=0.3.15",\
                "jaxlib>=0.3.15",\
                "numpy==1.23.1",\
                ],\
        packages=find_packages(),\
        description = "Neuro/symbolic learning for biological sequences and strucutures"
    )


