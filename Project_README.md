**Subgroup Discovery in Textual Data Using Statistical Metadata and BERT Uncertainty Analysis**

This repository contains the code and experiments for a study on identifying problematic subgroups within textual datasets using subgroup discovery techniques originally designed for tabular data.
The project bridges Natural Language Processing (NLP) and exploratory data mining by converting unstructured text into structured metadata and analyzing model uncertainty across different text subgroups.



1. Text-to-Table Transformation
    Each text sample is processed to extract statistical features, such as:
    - Text length
    - Sentence and word counts
    - Token distribution patterns

2. Uncertainty Detection with BERT
    A BERT-based sequence classification model is trained on the target dataset.
    Prediction uncertainty is defined using a probability threshold p*, wherea prediction is marked uncertain if model confidence < p*

    The value of p* is chosen so that at least 10% of the test set is labeled as uncertain, ensuring a sufficiently large subset for subgroup analysis.

3. Subgroup Discovery with DivExplorer

   The resulting tabular dataset (metadata + uncertainty labels) is analyzed using the DivExplorer algorithm.
