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

![A short video overview of embcompare graphical user interface ](.assets/overview.webm)

# Installation

```bash
# basic install
pip install embcompare

# installation with the gui tool
pip install embcompare[gui]
```