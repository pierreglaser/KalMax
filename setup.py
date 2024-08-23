import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kalmax",  
    version="0.0.1", 
    author="Tom George",
    description="Kalman based neural decoding in Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomGeorge1234/KalMax", 
    packages=setuptools.find_packages(where='kalmax'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'jax', 'tqdm', 'ratinabox'
    ],
)