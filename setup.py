from setuptools import setup, find_packages

setup(
    name="gym-pusht",
    version="0.1.6",
    packages=find_packages(),
    install_requires=[
       "gymnasium",
       "opencv-python",
       "pygame",
       "pymunk",
       "scikit-image",
       "shapely"
    ],
    author="Rémi Cadène",
    author_email="re.cadene@gmail.com",
    description="pusht environment for gymnasium",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_package_name",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)