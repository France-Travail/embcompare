<div style="text-align: center;">
    <h2>EmbCompare</h2>
    <p style="font-weight: bold;">A simple python tool for embedding comparison </p>
</div>

EmbCompare is a small python package highly inspired by the [Embedding Comparator tool](https://github.com/mitvis/embedding-comparator) that helps you compare your embeddings both visually and numerically.

<div style="
    color: #93a1a1;
    background-color: #002b36;
    border: solid #073642 3px;
    border-radius: 1rem; 
    font-size: small;
    padding: 0.75rem;
    margin-top: 2rem;
    margin-bottom: 2rem;
">
EmbCompare keeps comparisons simples. All computations are made in memory and the package does not bring any embeddings storage management.

If you need a tool to store, track and compare your embedding generation experiments, you may like the [vectory](https://github.com/pentoai/vectory) project.
</div>

### Key features : 
- **Visual comparison** : GUI for comparison of two embeddings
- **Numerical comparison** : Calculation of comparison indicators between two embeddings for monitoring purposes

# Table of content

- [Table of content](#table-of-content)
- [Installation](#installation)
- [Usage](#usage)
  - [Config file](#config-file)
  - [JSON comparison report generation](#json-comparison-report-generation)
  - [GUI](#gui)

# Installation

```bash
# basic install
pip install embcompare

# installation with the gui tool
pip install embcompare[gui]
```

# Usage

EmbCompare provides a CLI with three sub-commands : 

- `embcompare add` is used to create or update a yaml file containing all embeddings infos : path, format, labels, term-frequencies, ... ;
- `embcompare report` is used to generate json reports containing comparison metrics ;
- `embcompare gui` is used to start a [streamlit](https://streamlit.io/) webapp to compare your embeddings visually.

## Config file

EmbCompare use a yaml file for referencing embeddings and relevant informations. By default, EmbCompare is looking
for a file named embcompare.yaml in the current working directory.

```yaml
embeddings: 
    first_embedding:
        name: My first embedding
        path: /abspath/to/firstembedding.json
        format: json
        frequencies: /abspath/to/freqs.json
        frequencies_format: json
        labels: /abspath/to/labels.pkl
        labels_format: pkl
    second_embedding:
        name: My second embedding
        path: /abspath/to/secondembedding.json
        format: word2vec
        frequencies: /abspath/to/freqs.pkl
        frequencies_format: pkl
        labels: /abspath/to/labels.json
        labels_format: json
```

The `embcompare add` command allow to update this file programatically (and even create it if it does not exist).

## JSON comparison report generation

EmbCompare aims to help to compare embedding thanks to numerical metrics that can be used to check if a new
generated embedding is very different from the last one. The command `embcompare report` can be used in two ways : 
- With a single embedding. In this case it generate a small report about the embedding : 
  ```bash
  embcompare report first_embedding
  # creates a first_embedding_report.json file containing some infos about the embedding
  ```
- With two embeddings. In this case it generate a comparison report about the two embeddings : 
  ```bash
  embcompare report first_embedding second_embedding
  # creates a first_embedding_second_embedding_report.json file containing comparison metrics
  ```

## GUI

![A short video overview of embcompare graphical user interface ](.assets/overview.webm)

The GUI is also very handy to compare embeddings. To start the GUI, use the commande `embcompare gui`. It will launch a streamlit app that will allow you to visually compare the embeddings you added in the configuration file.