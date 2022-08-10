from setuptools import setup

setup(\
        name="sepia", \
        packages=["sepia", "testing"], \
        version = "0.00000", \
        install_requires =[\
                "coverage==6.4.3",\
                "jax==0.3.15",\
                "jaxlib==0.3.15",\
                "numpy==1.23.1",\
                ] 
        description = "<<stealth-mode>>" \
    )


