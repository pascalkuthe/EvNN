
import os

from setuptools import setup


# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


VERSION = "0.1.0"
DESCRIPTION = "EVNN: a torch extension for custom event based RNN models."
AUTHOR = "TUD and RUB"
AUTHOR_EMAIL = "khaleelulla.khan_nazeer@tu-dresden.de"
URL = "https://tu-dresden.de/ing/elektrotechnik/iee/hpsn"
LICENSE = "Apache 2.0"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Libraries",
]
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
with open(f"frameworks/pytorch/_version.py", "wt") as f:
    f.write(f'__version__ = "{VERSION}"')

base_path = os.path.dirname(os.path.realpath(__file__))


setup(
    name="evnn_pytorch",
    version=VERSION,
    description=DESCRIPTION,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    keywords="pytorch machine learning rnn lstm gru custom op",
    packages=["evnn_pytorch"],
    package_dir={"evnn_pytorch": "frameworks/pytorch"},
    install_requires=[],
    classifiers=CLASSIFIERS,
)
