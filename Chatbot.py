# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/filereq/main.py
/kaggle/input/filereq/requirements.txt
/kaggle/input/mainfolder/readsmartbot.py
!pip install -r "/kaggle/input/filereq/requirements.txt"
Collecting langchain==0.0.284 (from -r /kaggle/input/filereq/requirements.txt (line 1))
  Downloading langchain-0.0.284-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: python-dotenv==1.0.0 in /opt/conda/lib/python3.10/site-packages (from -r /kaggle/input/filereq/requirements.txt (line 2)) (1.0.0)
Collecting streamlit==1.22.0 (from -r /kaggle/input/filereq/requirements.txt (line 3))
  Downloading streamlit-1.22.0-py2.py3-none-any.whl.metadata (7.3 kB)
Collecting unstructured==0.9.2 (from -r /kaggle/input/filereq/requirements.txt (line 4))
  Downloading unstructured-0.9.2-py3-none-any.whl.metadata (23 kB)
Collecting tiktoken==0.4.0 (from -r /kaggle/input/filereq/requirements.txt (line 5))
  Downloading tiktoken-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.2 kB)
Collecting faiss-cpu==1.7.4 (from -r /kaggle/input/filereq/requirements.txt (line 6))
  Downloading faiss_cpu-1.7.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.3 kB)
Collecting libmagic==1.0 (from -r /kaggle/input/filereq/requirements.txt (line 7))
  Downloading libmagic-1.0.tar.gz (3.7 kB)
  Preparing metadata (setup.py) ... - done
Collecting python-magic==0.4.27 (from -r /kaggle/input/filereq/requirements.txt (line 8))
  Downloading python_magic-0.4.27-py2.py3-none-any.whl.metadata (5.8 kB)
ERROR: Ignored the following versions that require a different python version: 0.55.2 Requires-Python <3.5
ERROR: Could not find a version that satisfies the requirement python-magic-bin==0.4.14 (from versions: none)
ERROR: No matching distribution found for python-magic-bin==0.4.14
! pip install -qq -U sentence_transformers==2.2.2 langchain tiktoken pypdf faiss-gpu InstructorEmbedding transformers accelerate bitsandbytes streamlit==1.22.0
!pip install openai langchain langchain-openai
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cudf 23.8.0 requires cubinlinker, which is not installed.
cudf 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
cudf 23.8.0 requires ptxcompiler, which is not installed.
cuml 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
dask-cudf 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
keras-cv 0.8.2 requires keras-core, which is not installed.
keras-nlp 0.9.3 requires keras-core, which is not installed.
tensorflow-decision-forests 1.8.1 requires wurlitzer, which is not installed.
apache-beam 2.46.0 requires dill<0.3.2,>=0.3.1.1, but you have dill 0.3.8 which is incompatible.
apache-beam 2.46.0 requires numpy<1.25.0,>=1.14.3, but you have numpy 1.26.4 which is incompatible.
apache-beam 2.46.0 requires pyarrow<10.0.0,>=3.0.0, but you have pyarrow 15.0.2 which is incompatible.
cudf 23.8.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.4.0 which is incompatible.
cudf 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
cudf 23.8.0 requires protobuf<5,>=4.21, but you have protobuf 3.20.3 which is incompatible.
cudf 23.8.0 requires pyarrow==11.*, but you have pyarrow 15.0.2 which is incompatible.
cuml 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
dask-cuda 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
dask-cuda 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
dask-cudf 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
dask-cudf 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
distributed 2023.7.1 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
google-cloud-bigquery 2.34.4 requires packaging<22.0dev,>=14.3, but you have packaging 23.2 which is incompatible.
jupyterlab 4.1.6 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
jupyterlab-lsp 5.1.0 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
libpysal 4.9.2 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
momepy 0.7.0 requires shapely>=2, but you have shapely 1.8.5.post1 which is incompatible.
osmnx 1.9.2 requires shapely>=2.0, but you have shapely 1.8.5.post1 which is incompatible.
raft-dask 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
spopt 0.6.0 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
tensorflow 2.15.0 requires keras<2.16,>=2.15.0, but you have keras 3.2.1 which is incompatible.
ydata-profiling 4.6.4 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.
Collecting openai
  Downloading openai-1.24.0-py3-none-any.whl.metadata (21 kB)
Requirement already satisfied: langchain in /opt/conda/lib/python3.10/site-packages (0.1.16)
Collecting langchain-openai
  Downloading langchain_openai-0.1.4-py3-none-any.whl.metadata (2.5 kB)
Requirement already satisfied: anyio<5,>=3.5.0 in /opt/conda/lib/python3.10/site-packages (from openai) (4.2.0)
Requirement already satisfied: distro<2,>=1.7.0 in /opt/conda/lib/python3.10/site-packages (from openai) (1.9.0)
Requirement already satisfied: httpx<1,>=0.23.0 in /opt/conda/lib/python3.10/site-packages (from openai) (0.27.0)
Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/conda/lib/python3.10/site-packages (from openai) (2.5.3)
Requirement already satisfied: sniffio in /opt/conda/lib/python3.10/site-packages (from openai) (1.3.0)
Requirement already satisfied: tqdm>4 in /opt/conda/lib/python3.10/site-packages (from openai) (4.66.1)
Requirement already satisfied: typing-extensions<5,>=4.7 in /opt/conda/lib/python3.10/site-packages (from openai) (4.9.0)
Requirement already satisfied: PyYAML>=5.3 in /opt/conda/lib/python3.10/site-packages (from langchain) (6.0.1)
Requirement already satisfied: SQLAlchemy<3,>=1.4 in /opt/conda/lib/python3.10/site-packages (from langchain) (2.0.25)
Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /opt/conda/lib/python3.10/site-packages (from langchain) (3.9.1)
Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /opt/conda/lib/python3.10/site-packages (from langchain) (4.0.3)
Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /opt/conda/lib/python3.10/site-packages (from langchain) (0.6.4)
Requirement already satisfied: jsonpatch<2.0,>=1.33 in /opt/conda/lib/python3.10/site-packages (from langchain) (1.33)
Requirement already satisfied: langchain-community<0.1,>=0.0.32 in /opt/conda/lib/python3.10/site-packages (from langchain) (0.0.34)
Requirement already satisfied: langchain-core<0.2.0,>=0.1.42 in /opt/conda/lib/python3.10/site-packages (from langchain) (0.1.46)
Requirement already satisfied: langchain-text-splitters<0.1,>=0.0.1 in /opt/conda/lib/python3.10/site-packages (from langchain) (0.0.1)
Requirement already satisfied: langsmith<0.2.0,>=0.1.17 in /opt/conda/lib/python3.10/site-packages (from langchain) (0.1.52)
Requirement already satisfied: numpy<2,>=1 in /opt/conda/lib/python3.10/site-packages (from langchain) (1.26.4)
Requirement already satisfied: requests<3,>=2 in /opt/conda/lib/python3.10/site-packages (from langchain) (2.31.0)
Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /opt/conda/lib/python3.10/site-packages (from langchain) (8.2.3)
Requirement already satisfied: tiktoken<1,>=0.5.2 in /opt/conda/lib/python3.10/site-packages (from langchain-openai) (0.6.0)
Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (23.2.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.0.4)
Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.9.3)
Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.4.1)
Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
Requirement already satisfied: idna>=2.8 in /opt/conda/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai) (3.6)
Requirement already satisfied: exceptiongroup>=1.0.2 in /opt/conda/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai) (1.2.0)
Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /opt/conda/lib/python3.10/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (3.21.1)
Requirement already satisfied: typing-inspect<1,>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (0.9.0)
Requirement already satisfied: certifi in /opt/conda/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai) (2024.2.2)
Requirement already satisfied: httpcore==1.* in /opt/conda/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai) (1.0.5)
Requirement already satisfied: h11<0.15,>=0.13 in /opt/conda/lib/python3.10/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
Requirement already satisfied: jsonpointer>=1.9 in /opt/conda/lib/python3.10/site-packages (from jsonpatch<2.0,>=1.33->langchain) (2.4)
Requirement already satisfied: packaging<24.0,>=23.2 in /opt/conda/lib/python3.10/site-packages (from langchain-core<0.2.0,>=0.1.42->langchain) (23.2)
Requirement already satisfied: orjson<4.0.0,>=3.9.14 in /opt/conda/lib/python3.10/site-packages (from langsmith<0.2.0,>=0.1.17->langchain) (3.10.1)
Requirement already satisfied: annotated-types>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai) (0.6.0)
Requirement already satisfied: pydantic-core==2.14.6 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai) (2.14.6)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2->langchain) (3.3.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2->langchain) (1.26.18)
Requirement already satisfied: greenlet!=0.4.17 in /opt/conda/lib/python3.10/site-packages (from SQLAlchemy<3,>=1.4->langchain) (3.0.3)
Requirement already satisfied: regex>=2022.1.18 in /opt/conda/lib/python3.10/site-packages (from tiktoken<1,>=0.5.2->langchain-openai) (2023.12.25)
Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/conda/lib/python3.10/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain) (1.0.0)
Downloading openai-1.24.0-py3-none-any.whl (312 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 312.3/312.3 kB 2.7 MB/s eta 0:00:00
Downloading langchain_openai-0.1.4-py3-none-any.whl (33 kB)
Installing collected packages: openai, langchain-openai
Successfully installed langchain-openai-0.1.4 openai-1.24.0
!pip install unstructured==0.5.6
Collecting unstructured==0.5.6
  Downloading unstructured-0.5.6.tar.gz (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 6.5 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... - done
Collecting argilla (from unstructured==0.5.6)
  Downloading argilla-1.27.0-py3-none-any.whl.metadata (17 kB)
Requirement already satisfied: lxml in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (5.2.1)
Requirement already satisfied: nltk in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (3.2.4)
Requirement already satisfied: openpyxl in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (3.1.2)
Requirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (2.1.4)
Requirement already satisfied: pillow in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (9.5.0)
Collecting pypandoc (from unstructured==0.5.6)
  Downloading pypandoc-1.13-py3-none-any.whl.metadata (16 kB)
Collecting python-docx (from unstructured==0.5.6)
  Downloading python_docx-1.1.1-py3-none-any.whl.metadata (2.0 kB)
Collecting python-pptx (from unstructured==0.5.6)
  Downloading python_pptx-0.6.23-py3-none-any.whl.metadata (18 kB)
Collecting python-magic (from unstructured==0.5.6)
  Using cached python_magic-0.4.27-py2.py3-none-any.whl.metadata (5.8 kB)
Requirement already satisfied: markdown in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (3.5.2)
Requirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (2.31.0)
Requirement already satisfied: certifi>=2022.12.07 in /opt/conda/lib/python3.10/site-packages (from unstructured==0.5.6) (2024.2.2)
Collecting httpx<=0.26,>=0.15 (from argilla->unstructured==0.5.6)
  Downloading httpx-0.26.0-py3-none-any.whl.metadata (7.6 kB)
Requirement already satisfied: deprecated~=1.2.0 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (1.2.14)
Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (23.2)
Requirement already satisfied: pydantic>=1.10.7 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (2.5.3)
Requirement already satisfied: wrapt<1.15,>=1.13 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (1.14.1)
Collecting numpy<1.24.0 (from argilla->unstructured==0.5.6)
  Downloading numpy-1.23.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
Requirement already satisfied: tqdm>=4.27.0 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (4.66.1)
Requirement already satisfied: backoff in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (2.2.1)
Collecting monotonic (from argilla->unstructured==0.5.6)
  Downloading monotonic-1.6-py2.py3-none-any.whl.metadata (1.5 kB)
Requirement already satisfied: rich!=13.1.0 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (13.7.0)
Requirement already satisfied: typer<0.10.0,>=0.6.0 in /opt/conda/lib/python3.10/site-packages (from argilla->unstructured==0.5.6) (0.9.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.10/site-packages (from pandas->unstructured==0.5.6) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas->unstructured==0.5.6) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.10/site-packages (from pandas->unstructured==0.5.6) (2023.4)
Requirement already satisfied: six in /opt/conda/lib/python3.10/site-packages (from nltk->unstructured==0.5.6) (1.16.0)
Requirement already satisfied: et-xmlfile in /opt/conda/lib/python3.10/site-packages (from openpyxl->unstructured==0.5.6) (1.1.0)
Collecting lxml (from unstructured==0.5.6)
  Downloading lxml-4.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl.metadata (3.6 kB)
Requirement already satisfied: typing-extensions>=4.9.0 in /opt/conda/lib/python3.10/site-packages (from python-docx->unstructured==0.5.6) (4.9.0)
Collecting XlsxWriter>=0.5.7 (from python-pptx->unstructured==0.5.6)
  Downloading XlsxWriter-3.2.0-py3-none-any.whl.metadata (2.6 kB)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->unstructured==0.5.6) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->unstructured==0.5.6) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->unstructured==0.5.6) (1.26.18)
Requirement already satisfied: anyio in /opt/conda/lib/python3.10/site-packages (from httpx<=0.26,>=0.15->argilla->unstructured==0.5.6) (4.2.0)
Requirement already satisfied: httpcore==1.* in /opt/conda/lib/python3.10/site-packages (from httpx<=0.26,>=0.15->argilla->unstructured==0.5.6) (1.0.5)
Requirement already satisfied: sniffio in /opt/conda/lib/python3.10/site-packages (from httpx<=0.26,>=0.15->argilla->unstructured==0.5.6) (1.3.0)
Requirement already satisfied: h11<0.15,>=0.13 in /opt/conda/lib/python3.10/site-packages (from httpcore==1.*->httpx<=0.26,>=0.15->argilla->unstructured==0.5.6) (0.14.0)
Requirement already satisfied: annotated-types>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from pydantic>=1.10.7->argilla->unstructured==0.5.6) (0.6.0)
Requirement already satisfied: pydantic-core==2.14.6 in /opt/conda/lib/python3.10/site-packages (from pydantic>=1.10.7->argilla->unstructured==0.5.6) (2.14.6)
Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/conda/lib/python3.10/site-packages (from rich!=13.1.0->argilla->unstructured==0.5.6) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/conda/lib/python3.10/site-packages (from rich!=13.1.0->argilla->unstructured==0.5.6) (2.17.2)
Requirement already satisfied: click<9.0.0,>=7.1.1 in /opt/conda/lib/python3.10/site-packages (from typer<0.10.0,>=0.6.0->argilla->unstructured==0.5.6) (8.1.7)
Requirement already satisfied: mdurl~=0.1 in /opt/conda/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich!=13.1.0->argilla->unstructured==0.5.6) (0.1.2)
Requirement already satisfied: exceptiongroup>=1.0.2 in /opt/conda/lib/python3.10/site-packages (from anyio->httpx<=0.26,>=0.15->argilla->unstructured==0.5.6) (1.2.0)
Downloading argilla-1.27.0-py3-none-any.whl (420 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 420.8/420.8 kB 13.1 MB/s eta 0:00:00
Downloading pypandoc-1.13-py3-none-any.whl (21 kB)
Downloading python_docx-1.1.1-py3-none-any.whl (242 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 242.8/242.8 kB 13.8 MB/s eta 0:00:00
Downloading lxml-4.9.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (7.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.1/7.1 MB 40.2 MB/s eta 0:00:00
Downloading python_magic-0.4.27-py2.py3-none-any.whl (13 kB)
Downloading python_pptx-0.6.23-py3-none-any.whl (471 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 471.6/471.6 kB 10.5 MB/s eta 0:00:00
Downloading httpx-0.26.0-py3-none-any.whl (75 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.9/75.9 kB 772.5 kB/s eta 0:00:00
Downloading numpy-1.23.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.1/17.1 MB 41.5 MB/s eta 0:00:00
Downloading XlsxWriter-3.2.0-py3-none-any.whl (159 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 159.9/159.9 kB 9.4 MB/s eta 0:00:00
Downloading monotonic-1.6-py2.py3-none-any.whl (8.2 kB)
Building wheels for collected packages: unstructured
  Building wheel for unstructured (setup.py) ... - \ done
  Created wheel for unstructured: filename=unstructured-0.5.6-py3-none-any.whl size=1315700 sha256=3c0a2eac585b3245bdc9dd17f0213454061b2df3ce477cdb5dadcf066bdb88e9
  Stored in directory: /root/.cache/pip/wheels/8d/ba/38/39adebfeae4a6d48b6851964610929b716c66c861f7f6404f4
Successfully built unstructured
Installing collected packages: monotonic, XlsxWriter, python-magic, pypandoc, numpy, lxml, python-pptx, python-docx, httpx, argilla, unstructured
  Attempting uninstall: numpy
    Found existing installation: numpy 1.26.4
    Uninstalling numpy-1.26.4:
      Successfully uninstalled numpy-1.26.4
  Attempting uninstall: lxml
    Found existing installation: lxml 5.2.1
    Uninstalling lxml-5.2.1:
      Successfully uninstalled lxml-5.2.1
  Attempting uninstall: httpx
    Found existing installation: httpx 0.27.0
    Uninstalling httpx-0.27.0:
      Successfully uninstalled httpx-0.27.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cudf 23.8.0 requires cubinlinker, which is not installed.
cudf 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
cudf 23.8.0 requires ptxcompiler, which is not installed.
cuml 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
dask-cudf 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
keras-nlp 0.9.3 requires keras-core, which is not installed.
tensorflow-decision-forests 1.8.1 requires wurlitzer, which is not installed.
albumentations 1.4.0 requires numpy>=1.24.4, but you have numpy 1.23.5 which is incompatible.
apache-beam 2.46.0 requires dill<0.3.2,>=0.3.1.1, but you have dill 0.3.8 which is incompatible.
apache-beam 2.46.0 requires pyarrow<10.0.0,>=3.0.0, but you have pyarrow 15.0.2 which is incompatible.
chex 0.1.86 requires numpy>=1.24.1, but you have numpy 1.23.5 which is incompatible.
cudf 23.8.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.4.0 which is incompatible.
cudf 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
cudf 23.8.0 requires protobuf<5,>=4.21, but you have protobuf 3.20.3 which is incompatible.
cudf 23.8.0 requires pyarrow==11.*, but you have pyarrow 15.0.2 which is incompatible.
cuml 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
dask-cuda 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
dask-cuda 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
dask-cudf 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
dask-cudf 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
featuretools 1.30.0 requires numpy>=1.25.0, but you have numpy 1.23.5 which is incompatible.
jupyterlab 4.1.6 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
jupyterlab-lsp 5.1.0 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
libpysal 4.9.2 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
momepy 0.7.0 requires shapely>=2, but you have shapely 1.8.5.post1 which is incompatible.
osmnx 1.9.2 requires shapely>=2.0, but you have shapely 1.8.5.post1 which is incompatible.
pyldavis 3.4.1 requires numpy>=1.24.2, but you have numpy 1.23.5 which is incompatible.
pylibraft 23.8.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.4.0 which is incompatible.
raft-dask 23.8.0 requires dask==2023.7.1, but you have dask 2024.4.1 which is incompatible.
rmm 23.8.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.4.0 which is incompatible.
spopt 0.6.0 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
tensorflow 2.15.0 requires keras<2.16,>=2.15.0, but you have keras 3.2.1 which is incompatible.
tensorstore 0.1.56 requires ml-dtypes>=0.3.1, but you have ml-dtypes 0.2.0 which is incompatible.
woodwork 0.30.0 requires numpy>=1.25.0, but you have numpy 1.23.5 which is incompatible.
Successfully installed XlsxWriter-3.2.0 argilla-1.27.0 httpx-0.26.0 lxml-4.9.2 monotonic-1.6 numpy-1.23.5 pypandoc-1.13 python-docx-1.1.1 python-magic-0.4.27 python-pptx-0.6.23 unstructured-0.5.6
!wget -q -O - icanhazip.com
104.198.22.73
!yes | streamlit run "/kaggle/input/mainfolder/readsmartbot.py" & yes | npx localtunnel --port 8501
npm WARN exec The following package was not found and will be installed: localtunnel@2.0.2

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.


  You can now view your Streamlit app in your browser.

  Network URL: http://172.19.2.2:8501
  External URL: http://104.198.22.73:8501

your url is: https://six-radios-return.loca.lt
/root/.npm/_npx/75ac80b86e83d4a2/node_modules/localtunnel/bin/lt.js:81
    throw err;
    ^

Error: connection refused: localtunnel.me:44249 (check your firewall settings)
    at Socket.<anonymous> (/root/.npm/_npx/75ac80b86e83d4a2/node_modules/localtunnel/lib/TunnelCluster.js:52:11)
    at Socket.emit (node:events:514:28)
    at emitErrorNT (node:internal/streams/destroy:151:8)
    at emitErrorCloseNT (node:internal/streams/destroy:116:3)
    at process.processTicksAndRejections (node:internal/process/task_queues:82:21)

Node.js v20.9.0
npm notice 
npm notice New minor version of npm available! 10.1.0 -> 10.6.0
npm notice Changelog: https://github.com/npm/cli/releases/tag/v10.6.0
npm notice Run npm install -g npm@10.6.0 to update!
npm notice 
yes: standard output: Broken pipe