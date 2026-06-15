# Trained models

The fine-tuned model weights are not included in this repository because of their size.

The extraction pipeline expects the selected PubMedBERT model at:

`models/pubmedbert_chia_ner_li_nontest1900_v1/`

The model was fine-tuned for token classification on the CHIA training data. Model-selection results and training details are reported in the thesis.

To run Branch A, provide the local model directory through the `--model-dir` argument.