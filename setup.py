from pathlib import Path
from setuptools import setup, find_packages

# Project metadata
NAME = "hcse"                       # pip install NAME
VERSION = "0.1.4"                   # update on each release
DESCRIPTION = "Holland-Consciousness-Scaling-Engine (HCSE)"
URL = "https://github.com/ayjays132/Holland-Consciousness-Scaling-Engine-HCSE"
AUTHOR = "Phillip Holland"
AUTHOR_EMAIL = "ayjays.ph@gmail.com"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.8"

# Long description from README.md
root = Path(__file__).parent
LONG_DESC = (root / "README.md").read_text(encoding="utf-8")

# Runtime requirements (sync with requirements-dev.txt if needed)
INSTALL_REQUIRES = [
    # "torch>=2.0",           # example – uncomment / edit as required
    # "numpy>=1.24",
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    packages=find_packages(
        exclude=["tests*", "docs*", "examples*", "*.github*"]
    ),
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # ship non-code files declared in MANIFEST.in
)
