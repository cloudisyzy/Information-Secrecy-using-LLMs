# Original imports and setup
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers import T5Tokenizer, T5ForConditionalGeneration
from IPython.display import display, HTML
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *  # Assuming helper functions like masking, add_noise_with_snr, etc., are defined here
import json
import numpy as np

def main():
    # Automatically choose (prefer NVIDIA GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify model name
    model_name = "facebook/bart-base"
    tokenizer_bart = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Specify model name for the summarizer
    summarizer_name = "Falconsai/text_summarization"
    tokenizer_sum = T5Tokenizer.from_pretrained(summarizer_name)
    summarizer = T5ForConditionalGeneration.from_pretrained(summarizer_name).to(device)

    # File path for the JSONL file
    file_path = r"E:\info_project\Information-Secrecy-using-LLMs\dataset\3_samples.jsonl"  # Replace with your actual file path
    texts = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
    print(texts)
    # Define parameters
    nr_rounds = 5
    snr_range = list(range(-10, 40, 5))

    # Storage for results
    all_cs_list_mean = []
    all_cs_list_lower = []
    all_cs_list_upper = []
    all_mi_list_mean = []
    all_mi_list_lower = []
    all_mi_list_upper = []

    # Process each text sample
    for text_index, text in enumerate(tqdm(texts, desc="Processing Samples")):
        input_text_ls = [masking(text, 4 / len(text)) for _ in range(nr_rounds)]

        # Generate baseline summaries and embeddings for the current text
        em_baseline_summary = []
        for i in tqdm(range(nr_rounds), desc=f"Baseline Processing for Sample {text_index + 1}"):
            input_ids = tokenizer_bart(input_text_ls[i], return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                encoder_outputs = model.model.encoder(input_ids=input_ids)

            baseline_outputs = model.generate(input_ids=None, encoder_outputs=encoder_outputs, max_length=300, min_length=100,
                                              num_beams=15, do_sample=True, temperature=0.15, early_stopping=True)
            baseline_text = tokenizer_bart.decode(baseline_outputs[0], skip_special_tokens=True)
            baseline_ids = tokenizer_sum(baseline_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                baseline_encoder_outputs = summarizer.encoder(input_ids=baseline_ids)
            baseline_sum_output = summarizer.generate(input_ids=None, encoder_outputs=baseline_encoder_outputs, max_length=70,
                                                      output_hidden_states=True, return_dict_in_generate=True,
                                                      do_sample=True, temperature=0.1)
            em_baseline_summary.append(extract_hidden_states(baseline_sum_output.decoder_hidden_states))

        # Process noisy embeddings and calculate metrics for the current text
        cs_list_mean = []
        cs_list_lower = []
        cs_list_upper = []
        mi_list_mean = []
        mi_list_lower = []
        mi_list_upper = []

        for target_snr in tqdm(snr_range, desc=f"SNR Range for Sample {text_index + 1}"):
            cs_list_texts = []
            mi_list_texts = []
            for i in range(nr_rounds):
                input_ids = tokenizer_bart(input_text_ls[i], return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    encoder_outputs = model.model.encoder(input_ids=input_ids)

                noisy_encoder_output = add_noise_with_snr(encoder_outputs.last_hidden_state, "gaussian", target_snr, 0.4, 0.4)
                modified_encoder_outputs = BaseModelOutput(last_hidden_state=noisy_encoder_output)

                noisy_outputs = model.generate(input_ids=None, encoder_outputs=modified_encoder_outputs, max_length=300, min_length=100,
                                               num_beams=15, do_sample=True, temperature=0.15, early_stopping=True)
                noisy_text = tokenizer_bart.decode(noisy_outputs[0], skip_special_tokens=True)
                noisy_ids = tokenizer_sum(noisy_text, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    noisy_encoder_outputs = summarizer.encoder(input_ids=noisy_ids)
                noisy_sum_output = summarizer.generate(input_ids=None, encoder_outputs=noisy_encoder_outputs, max_length=70,
                                                       output_hidden_states=True, return_dict_in_generate=True,
                                                       do_sample=True, temperature=0.1)
                em_noisy_summary = extract_hidden_states(noisy_sum_output.decoder_hidden_states)

                # Align tensors
                pad_em_baseline_summary, pad_em_noisy_summary = align_tensors(em_baseline_summary[i], em_noisy_summary)

                # Calculate metrics
                cs_list_texts.append(F.cosine_similarity(pad_em_baseline_summary, pad_em_noisy_summary, dim=1).mean().item())
                em_noisy_summary_np = em_noisy_summary.cpu().numpy()
                em_baseline_summary_np = em_baseline_summary[i].cpu().numpy()
                mi_list_texts.append(ksg(em_baseline_summary_np, em_noisy_summary_np))

            # Aggregate results for the current SNR
            cs_list_mean.append(np.mean(cs_list_texts))
            mi_list_mean.append(np.mean(mi_list_texts))

            z = 1.96  # For 95% confidence interval
            cs_list_std = np.std(cs_list_texts, axis=0)
            margin_of_error = z * (cs_list_std / np.sqrt(nr_rounds))
            cs_list_lower.append(cs_list_mean[-1] - margin_of_error)
            cs_list_upper.append(cs_list_mean[-1] + margin_of_error)

            mi_list_std = np.std(mi_list_texts, axis=0)
            margin_of_error = z * (mi_list_std / np.sqrt(nr_rounds))
            mi_list_lower.append(mi_list_mean[-1] - margin_of_error)
            mi_list_upper.append(mi_list_mean[-1] + margin_of_error)

        # Store results for the current text sample
        all_cs_list_mean.append(cs_list_mean)
        all_cs_list_lower.append(cs_list_lower)
        all_cs_list_upper.append(cs_list_upper)
        all_mi_list_mean.append(mi_list_mean)
        all_mi_list_lower.append(mi_list_lower)
        all_mi_list_upper.append(mi_list_upper)

    # Plot results for all text samples
    plt.figure(figsize=(20, 6))
    for i in range(len(texts)):
        plt.plot(snr_range, all_cs_list_mean[i], label=f"Text Sample {i + 1}")
        plt.fill_between(snr_range, all_cs_list_lower[i], all_cs_list_upper[i], alpha=0.2)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity vs SNR for Multiple Text Samples")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 6))
    for i in range(len(texts)):
        plt.plot(snr_range, all_mi_list_mean[i], label=f"Text Sample {i + 1}")
        plt.fill_between(snr_range, all_mi_list_lower[i], all_mi_list_upper[i], alpha=0.2)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Mutual Information")
    plt.title("Mutual Information vs SNR for Multiple Text Samples")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
