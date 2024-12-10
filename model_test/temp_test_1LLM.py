# for first LLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# for second
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

from IPython.display import display, HTML
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# importing all helper functions
from utils import *

def main():
    # Automatically choose (prefer NVIDIA GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify model name
    model_name = "facebook/bart-base"
    # model_name = "facebook/bart-large" # Recommend this one if your computer is okay with larger models

    tokenizer_bart = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Specify model name
    summarizer_name = "Falconsai/text_summarization"

    tokenizer_sum = T5Tokenizer.from_pretrained(summarizer_name)
    summarizer = T5ForConditionalGeneration.from_pretrained(summarizer_name).to(device)

    original_text = """
    Beginners BBQ Class Taking Place in Missoula! 
    Do you want to get better at making delicious BBQ?
    You will have the opportunity, put this on your calendar now. 
    Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. 
    He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. 
    He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. 
    The cost to be in the class is $35 per person, and for spectators it is free. 
    Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.
    """.replace("\n", "")

    input_text_ls = []
    nr_rounds = 10

    for _ in range(nr_rounds):
        input_text_ls.append(masking(original_text, 4/130))

    original_ids = tokenizer_sum('Please summarize: ' + original_text, return_tensors="pt").input_ids.to(device)
    ### Note, it seems 'Please Summarize: ' somehow leads to more deterministic summarization

    with torch.no_grad():
        original_encoder_outputs = summarizer.encoder(input_ids=original_ids)
        
    original_sum_output = summarizer.generate(input_ids=None, encoder_outputs=original_encoder_outputs, max_length=70, output_hidden_states=True,
                                            return_dict_in_generate=True, do_sample=True, num_beams=1, temperature=0.1)
    original_summary = tokenizer_sum.decode(original_sum_output.sequences[0], skip_special_tokens=True)

    em_original_summary = extract_hidden_states(original_sum_output.decoder_hidden_states)

    
    # Define temperature range

    
    temp_range = [0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3] 
    cs_base_orig_mean = []
    cs_base_orig_lower = []
    cs_base_orig_upper = []
    mi_base_orig_mean = []
    mi_base_orig_lower = []
    mi_base_orig_upper = []
    
    cs_base_noisy_mean = []
    cs_base_noisy_lower = []
    cs_base_noisy_upper = []
    mi_base_noisy_mean = []
    mi_base_noisy_lower = []
    mi_base_noisy_upper = []


    for target_temp in tqdm(temp_range):
        cs_list_base_orig = []
        mi_list_base_orig = []
        cs_list_base_noisy = []
        mi_list_base_noisy = []
        
        for i in tqdm(range(nr_rounds)):
            
            ## Baseline
            input_ids_base = tokenizer_bart(input_text_ls[i], return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                encoder_outputs = model.model.encoder(input_ids=input_ids_base)
                
            # first LLM
            baseline_outputs = model.generate(input_ids=None, encoder_outputs=encoder_outputs, max_length=300, min_length=100, 
                                            num_beams=1, do_sample=True, temperature=target_temp, early_stopping=True)
            baseline_text = tokenizer_bart.decode(baseline_outputs[0], skip_special_tokens=True)
            # print(baseline_text)
        
            
            baseline_ids = tokenizer_sum('Please summarize: ' + baseline_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                baseline_encoder_outputs = summarizer.encoder(input_ids=baseline_ids)
                
            baseline_sum_output = summarizer.generate(input_ids=None, encoder_outputs=baseline_encoder_outputs, max_length=70, output_hidden_states=True,
                                                    return_dict_in_generate=True, do_sample=True, num_beams=1, temperature=0.1)
            baseline_summary = tokenizer_sum.decode(baseline_sum_output.sequences[0], skip_special_tokens=True)
            # print(baseline_summary)
            em_baseline_summary = extract_hidden_states(baseline_sum_output.decoder_hidden_states)
            
            # compare to original
            # pad
            pad_em_baseline_summary, pad_em_original_summary = align_tensors(em_baseline_summary, em_original_summary)
            
            # calculate cs
            cs_list_base_orig.append(F.cosine_similarity(pad_em_baseline_summary, pad_em_original_summary, dim=1).mean().item())
        
            # calculate mi
            em_original_summary_np = em_original_summary.cpu().numpy()
            em_baseline_summary_np = em_baseline_summary.cpu().numpy()
            mi_list_base_orig.append(ksg(em_baseline_summary_np, em_original_summary_np))
            
            
            ## Noisy
            
            input_ids_noisy = tokenizer_bart(input_text_ls[i], return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                encoder_outputs = model.model.encoder(input_ids=input_ids_noisy)
            
            # add noise
            noisy_encoder_output = add_noise_with_snr(
                encoder_output = encoder_outputs.last_hidden_state,
                noise_type = "gaussian",
                target_snr_db = 10,
                dropout_rate=0.4,
                sp_thresh=0.4
            )
            modified_encoder_outputs = BaseModelOutput(last_hidden_state=noisy_encoder_output)
            
            # first LLM
            noisy_outputs = model.generate(input_ids=None, encoder_outputs=modified_encoder_outputs, max_length=300, min_length=100, 
                                        num_beams=1, do_sample=True, temperature=target_temp, early_stopping=True)
            noisy_text = tokenizer_bart.decode(noisy_outputs[0], skip_special_tokens=True)
            #print(noisy_text)
        
            # second LLM
            noisy_ids = tokenizer_sum('Please summarize: ' + noisy_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                noisy_encoder_outputs = summarizer.encoder(input_ids=noisy_ids)
            noisy_sum_output = summarizer.generate(input_ids=None, encoder_outputs=noisy_encoder_outputs, max_length=70, output_hidden_states=True,
                                                return_dict_in_generate=True, do_sample=True, num_beams=1, temperature=0.1)
            noisy_summary = tokenizer_sum.decode(noisy_sum_output.sequences[0], skip_special_tokens=True)
            #print(noisy_summary)
            
            # get embeddings
            em_noisy_summary = extract_hidden_states(noisy_sum_output.decoder_hidden_states)
            
            # pad each embedding with embedding of baseline
            pad_em_baseline_summary, pad_em_noisy_summary = align_tensors(em_baseline_summary, em_noisy_summary)
            
            # calculate cs
            cs_list_base_noisy.append(F.cosine_similarity(pad_em_baseline_summary, pad_em_noisy_summary, dim=1).mean().item())
        
            # calculate mi
            em_noisy_summary_np = em_noisy_summary.cpu().numpy()
            em_baseline_summary_np = em_baseline_summary.cpu().numpy()
            mi_list_base_noisy.append(ksg(em_baseline_summary_np, em_noisy_summary_np))
            

            
        cs_base_orig_mean.append(np.mean(cs_list_base_orig))
        mi_base_orig_mean.append(np.mean(mi_list_base_orig))
                                 
        cs_base_noisy_mean.append(np.mean(cs_list_base_noisy))
        mi_base_noisy_mean.append(np.mean(mi_list_base_noisy))
        

        z = 1.96  # For 95% confidence level
        cs_list_std = np.std(cs_list_base_orig, axis=0)
        margin_of_error = z * (cs_list_std / np.sqrt(nr_rounds))
        cs_base_orig_lower.append(cs_base_orig_mean[-1] - margin_of_error)
        cs_base_orig_upper.append(cs_base_orig_mean[-1] + margin_of_error)

        mi_list_std = np.std(mi_list_base_orig, axis=0)
        margin_of_error = z * (mi_list_std / np.sqrt(nr_rounds))
        mi_base_orig_lower.append(mi_base_orig_mean[-1] - margin_of_error)
        mi_base_orig_upper.append(mi_base_orig_mean[-1] + margin_of_error)
        
        cs_list_std = np.std(cs_list_base_noisy, axis=0)
        margin_of_error = z * (cs_list_std / np.sqrt(nr_rounds))
        cs_base_noisy_lower.append(cs_base_noisy_mean[-1] - margin_of_error)
        cs_base_noisy_upper.append(cs_base_noisy_mean[-1] + margin_of_error)

        mi_list_std = np.std(mi_list_base_noisy, axis=0)
        margin_of_error = z * (mi_list_std / np.sqrt(nr_rounds))
        mi_base_noisy_lower.append(mi_base_noisy_mean[-1] - margin_of_error)
        mi_base_noisy_upper.append(mi_base_noisy_mean[-1] + margin_of_error)
        
    

    
    plt.figure(figsize=(20, 6), dpi=300)
    plt.plot(temp_range, cs_base_orig_mean, marker='o', label='between baseline and original')
    plt.fill_between(temp_range, cs_base_orig_lower, cs_base_orig_upper, color='b', alpha=0.2)
    plt.plot(temp_range, cs_base_noisy_mean, marker='o', label='between baseline and noisy')
    plt.fill_between(temp_range, cs_base_noisy_lower, cs_base_noisy_upper, color='r', alpha=0.2)
    plt.xlim([0.15,0.3])
    plt.ylim([0.0,1.1])
    plt.xlabel("Temperature")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Similarity for different Temperatures of LLM #1")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(20, 6), dpi=300)
    plt.plot(temp_range, mi_base_orig_mean, marker='o', label='between baseline and original')
    plt.fill_between(temp_range, mi_base_orig_lower, mi_base_orig_upper, color='b', alpha=0.2)
    plt.plot(temp_range, mi_base_noisy_mean, marker='o', label='between baseline and noisy')
    plt.fill_between(temp_range, mi_base_noisy_lower, mi_base_noisy_upper, color='r', alpha=0.2)
    plt.xlim([0.15,0.3])
    plt.ylim([0.0,5.1])
    plt.xlabel("Temperature")
    plt.ylabel("Mutual Information")
    plt.title("Mututal Information for different Temperatures of LLM #1")
    plt.grid()
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()