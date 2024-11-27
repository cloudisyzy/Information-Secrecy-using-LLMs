# Information Secrecy Using LLMs
**The project focuses on understading how adding noises to the latent representation of a transformer affects the performance of Encoder-Decoder LLMs**

- `main.py`: contains the execution code for the entire pipeline from getting the input text to generating latent representations, corrupting them, feeding the output into a summarizer and computing their similarity metrics.

- `utils.py`: contains the helper functions needed for running the pipeline like adding noise, masking the input text, extracting `last_hidden_state` outputs from the decoder, aligning tensors of unequal length and computing MI based similarity scores.

- `Experiment.ipynb`: is a Jupyter Notebook of the entire pipeline helpful for piecewise execution and debugging.


### Dependencies 

In order to calculate mutual information:
```
git clone https://github.com/gregversteeg/NPEET.git
cd NPEET
pip install .
```

***Authors:** Ziyue Yang, Boyue Jiang, Sreyan Ghosh, Theresa Hösl, Oscar Wink* \
***Supervisor:** Anubhab Ghosh*
