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
- create `.env` file based on [example.env](example.env)
- install requirements `pip install -r requirements.txt`
- extract the ˙`sumeczech` dataset into `./sumeczech-1.0/`


### Usage
- `python dataset_tool.py <input_json_file> <output_json_file>` 
- where `<input_json_file>` is the file with the data in the `.jsonl` format of the `SumeCzech` dataset
- and `<output_json_file>` is the file where the output will be saved in the `.jsonl` format



### Metacentrum

- ideálne nastavený login cez SSH klúč
- pripojenie s X11 forwarding
```bash
ssh -X zefron6.cerit-sc.cz
```

- qsub príkaz
```bash
qsub ./Jupyter_Cuda80_KNN.sh
```

---

## Base Evaluation
- used SumeCzech test dataset but only first 150 examples
- model: `Mistral-7B-Instruct-v0.2` but quantized to 8bit: `mistral-7b-instruct-v0.2.Q8_0.gguf`

--- 
### 0shot: abstract to headline

Given prompt: 
```
{"role": "user", "content": "Vytvoř stručný český nadpis, který výstižně shrnuje obsah tohoto abstraktu: \n '{abstract}'"},
```

### 3shot: abstract to headline

Given prompt: 
```
chat_abstract_to_headline_3shot = [
  {"role": "user", "content": "Jste užitečným pomocníkem, který shrne text v českém jazyce pro různé typy sumarizací.\n {abstract}"},
  {"role": "assistant", "content": {headline}},
  {"role": "user", "content": {abstract}},
  {"role": "assistant", "content": {headline}},
  {"role": "user", "content": {abstract}},
  {"role": "assistant", "content": {headline}},
  {"role": "user", "content": {abstract}},
]
```

### 3shot: text to abstract
- same prompt as above but abstract is text and headline is abstract

---

#### **RougeRAW**
- version of Rouge metric that does not use stemming or stop words removal
- used in SumeCzech paper.

>!**NOTE**: using high values from rougeraw metric 
This refers to the ROUGE-RAW score when the target summary length is set to the 90th percentile of the reference summary lengths. 

```
                  RougeRAW-1      RougeRAW-2      RougeRAW-L
                  P    R    F     P    R    F     P    R    F
0s-abs to hdln   12.9 22.6 16.4  03.1 06.1 04.0  11.3 19.8 14.1
3s-abs to hdln   13.7 23.6 16.6  03.5 06.8 04.4  11.8 21.2 14.7
3s-txt to abs    14.0 19.4 15.9  01.9 02.8 02.2  09.2 13.2 10.6
```

**Theirs**: <br> 
- abstract to headline

```
            RougeRAW-1      RougeRAW-2      RougeRAW-L
Method      P    R    F     P    R    F     P    R    F
first     13.9 23.6 16.5  04.1 07.4 05.0  12.2 20.7 14.5
random    11.0 17.8 12.8  02.6 04.5 03.1  09.6 15.5 11.1
textrank  13.3 22.8 15.9  03.7 06.8 04.6  11.6 19.9 13.8
t2t       20.2 15.9 17.2  06.7 05.1 05.6  18.6 14.7 15.8
```

- text to abstract
```
            RougeRAW-1      RougeRAW-2      RougeRAW-L
Method      P    R    F     P    R    F     P    R    F
first     13.1 17.9 14.4  01.9 02.8 02.1  08.8 12.0 09.6
random    11.7 15.5 12.7  01.2 01.7 01.3  07.7 10.3 08.4
textrank  11.1 20.8 13.8  01.6 03.1 02.0  07.1 13.4 08.9
t2t       13.2 10.5 11.3  01.2 00.9 01.0  10.2 08.1 08.7
```



#### **Rouge**
- Rouge metric from [Huggingface library](https://huggingface.co/spaces/evaluate-metric/rouge)
- without a stemmer
- showing only F1 scores


```
                Rouge-1    Rouge-2     Rouge-L    Rouge-Lsum
                  F         F             F           F
0s-abs to hdln   18.36      5.30        15.34       15.35
3s-abs to hdln   17.61      5.39        14.50       14.71
3s-txt to abs    24.53      4.65        13.67       13.87
```

#### **BertScore**
- uses model: `google-bert/bert-base-multilingual-cased` 

```
                    BertScore
                  P      R      F
0s-abs to hdln   0.659   0.692  0.675
3s-abs to hdln   0.657   0.693  0.674
3s-txt to abs    0.659   0.674  0.666
```

#### **BLEU**

```
                  BLEU 
0s-abs to hdln    0.011
3s-abs to hdln    0.008
3s-txt to abs     0.010
```

#### **METEOR**

```
                METEOR
0s-abs to hdln   0.125
3s-abs to hdln   0.133
3s-txt to abs    0.142
```

