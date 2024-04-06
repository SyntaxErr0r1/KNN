# KNN - text summarization 

This is the repository for KNN project at VUT FIT. 

As of now it consists of couple of Python Notebooks, which are used for data preprocessing and evaluation.


## Docs

Notes doc:
[https://docs.google.com/document/d/1VxYESr0JHd2iB6swp7XlGTmxkmW5QZ8Ofs7JtMrcwYg/edit#heading=h.k8mvfthgb23j]

Checkpoint doc:
[https://docs.google.com/document/d/1ENQ43E5LHeky2Nc-gqrUHDR-GbNUQfpfMnq47S-j0jo/edit#heading=h.wgl2s2eniag8]


### Development

Using virtual environment is recommended. But its slow afffff in venv. AAAAA

```bash
python3.11 -m venv .venv

# activate venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## dataset_tool.py
Tool for creating the dataset 

### Setup
- create .env file with following variables
- environment variables for using openai API
- install requirements
- extract the ˙`sumeczech` dataset into `./sumeczech-1.0/`


### Usage
- `python dataset_tool.py <input_json_file> <output_json_file>` 
- where `<input_json_file>` is the file with the data in the `.jsonl` format of the `SumeCzech` dataset
- and `<output_json_file>` is the file where the output will be saved in the `.jsonl` format