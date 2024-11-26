# Information-Secrecy-using-LLMs
The project focus on discover how adding noises to the latent representation (context vector) affects the performance of Encoder-Decoder LLMs

### Model Card

Here are some of the models I find interesting:

1. `google/flan-t5-small`  
   - Too small lead to too weak outputs, do not use it if not necessary.
   - **Avg VRAM alloc: 352.88 MB**
2. `google/flan-t5-base`  
   - Good one, it is small and the performance is not bad. Good at seq2seq tasks.
   - **Avg VRAM alloc: 1158.30 MB**
3. `google/flan-t5-large`  
   - A large one, but I think is feasible. Weird that the performance is very bad in translation task, even worse than the base one and the small one. Guess this is not trained for translation task.
   - **Avg VRAM alloc: 3176.94 MB**
4. `google/flan-t5-xl` 
   - Much more powerful than other small versions of flan-t5, but the computational cost is too high. Running on CPU is possible if having a PC with 32GB RAM.
   - **Avg VRAM alloc: 12515.98**
5. `Falconsai/text_summarization`
   - A candidate for our text summarizer, a fine-tuned t5-small model. Performs much better when using the transformers.pipeline
   - **Avg VRAM alloc: 280.51 MB**
6. `facebook/bart-base`
   - Good at filling in the blank task
   - **Avg VRAM alloc: 618.09 MB**
7. `facebook/bart-large`
   - Good at filling in the blank task, larger one.
   - **Avg VRAM alloc: 1710.58 MB**


