#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libraries

# In[1]:


from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from IPython.display import display, HTML
import torch


# ## 2. Choose Device (GPU / CPU)

# In[2]:


# Automatically choose (prefer NVIDIA GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # OR choose device manually, be sure to comment other codes relevant to `device`
# device = torch.device("cuda")
# device = torch.device("cpu")


# ## 3. Load BART model and tokenizer

# In[3]:


# Specify model name
# model_name = "facebook/bart-base"
model_name = "facebook/bart-large" # Recommend this one if your computer is okay with larger models

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)


# ## 4. A function adding AWGN noise to latent representation

# In[4]:


def add_noise_with_snr(encoder_output, target_snr_db):
    """
    Add noise to the encoder output based on a target SNR in dB.
    
    Parameters:
    - encoder_output: torch.Tensor, the encoder's output (last_hidden_state).
    - target_snr_db: float, the desired signal-to-noise ratio in dB.
    
    Returns:
    - noisy_encoder_output: torch.Tensor, encoder output with added noise.
    """
    # Convert SNR from dB to linear scale
    target_snr_linear = 10 ** (target_snr_db / 10)
    
    # Calculate power of the signal
    signal_power = torch.mean(encoder_output ** 2)
    
    # Calculate required noise power for the target SNR
    noise_power = signal_power / target_snr_linear
    noise = torch.randn_like(encoder_output) * torch.sqrt(noise_power)
    
    # Add noise to the encoder output
    noisy_encoder_output = encoder_output + noise
    return noisy_encoder_output


# ## 5. Three example encoder inputs for `fill in the blank` task
# 
# `original_text` contains the complete text <br>
# `input_text` contains the masked text

# In[5]:


original_text = """
Beginners BBQ Class Taking Place in Missoula! 
Do you want to get better at making delicious BBQ?
You will have the opportunity, put this on your calendar now. 
Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. 
He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. 
He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. 
The cost to be in the class is $35 per person, and for spectators it is free. 
Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.
"""
input_text = """
Beginners BBQ Class <mask> in Missoula! 
Do you want to <mask> making delicious BBQ?
You will have the opportunity, put this on your calendar now. 
Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. 
He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. 
He will teach you <mask> compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. 
The <mask> the class is $35 per person, and for spectators it is free. 
Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.
""".replace("\n", "")

# original_text = """
# This November, embark on an exciting hiking adventure! 
# Explore the scenic mountain trails with an experienced guide, who will show you the best routes and hidden viewpoints. 
# This journey is suitable for all levels, from beginners to advanced hikers. 
# The hike covers approximately 10 miles and includes multiple rest stops with breathtaking views. 
# Participants should bring water, snacks, and comfortable hiking shoes. 
# The cost of the trip is $60, which includes a map and a group photo.
# """
# input_text = """
# This November, embark on an exciting <mask> adventure! 
# Explore the scenic mountain trails with an experienced guide, who will show you the best routes and hidden <mask>. 
# This journey is suitable for all levels, from beginners to advanced <mask>. 
# The hike covers approximately 10 miles and includes multiple rest stops with breathtaking <mask>. 
# Participants should bring water, snacks, and comfortable hiking shoes. 
# The <mask> is $60, which includes a map and a group photo.
# """.replace("\n", "")

# original_text = """
# Welcome to our online coding bootcamp program! 
# Whether you're a complete beginner or looking to improve your programming skills, this course is designed for you. 
# Throughout the course, you will learn essential coding languages such as Python and JavaScript. 
# Our instructors will guide you through interactive projects and provide real-time feedback. 
# Each student will receive a certificate of completion at the end of the program. 
# The total cost for the bootcamp is $150, which includes all learning materials.
# """
# input_text = """
# Welcome to our online <mask> bootcamp program! 
# Whether you're a complete beginner or looking to <mask> your programming skills, this course is designed for you. 
# Throughout the course, you will learn essential <mask> such as Python and JavaScript. 
# Our instructors will guide you through interactive projects and provide real-time <mask>. 
# Each student will receive a certificate of completion at the end of the <mask>. 
# The total cost for the bootcamp is $150, which <mask> all learning materials.
# """.replace("\n", "")


# ## 6. Pass the `input_text` through the LLM

# ### 6.1 Tokenize the `input_text` to tokens (integer numbers)

# In[6]:


input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)


# ### 6.2 Get encoder output

# In[7]:


with torch.no_grad():
    encoder_outputs = model.model.encoder(input_ids=input_ids)


# ### 6.3.1 Case 1: clean latent reprenstation (without noise)

# In[8]:


# # Generate output with the clean encoder output (latent reprenstation)
baseline_outputs = model.generate(
    input_ids=None,                   # No input tokens are provided here, as we're feeding encoder outputs directly
    encoder_outputs=encoder_outputs,  # Encoded representations from the encoder
    max_length=200,                   # Set maximum length for the generated text sequence
    min_length=10,                    # Set minimum length for the generated text sequence
    do_sample=True,                   # Enables sampling for diverse outputs, rather than greedy decoding
    temperature=0.1                   # Low temperature to control randomness, resulting in less varied output
)

# Decode the decoder output using tokenizer
baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)


# ### 6.3.2 Case 2: noisy latent reprenstation

# In[9]:


# Add noise with a target SNR and generate noisy output (latent reprenstation)
target_snr = 3  # Set target SNR
noisy_encoder_output = add_noise_with_snr(encoder_outputs.last_hidden_state, target_snr) # Add noise
modified_encoder_outputs = BaseModelOutput(last_hidden_state=noisy_encoder_output) # There are slight differences between `noisy_encoder_output` and `modified_encoder_outputs`, print them for more info

# Generate output with the noisy encoder output
noisy_outputs = model.generate(
    input_ids=None,                            # No input tokens are provided here, as we're feeding encoder outputs directly
    encoder_outputs=modified_encoder_outputs,  # Encoded representations from the encoder
    max_length=200,                            # Set maximum length for the generated text sequence
    min_length=10,                             # Set minimum length for the generated text sequence
    do_sample=True,                            # Enables sampling for diverse outputs, rather than greedy decoding
    temperature=0.1                            # Low temperature to control randomness, resulting in less varied output
)

# Decode the decoder output using tokenizer
noisy_text = tokenizer.decode(noisy_outputs[0], skip_special_tokens=True)


# ## 7. Display original texts and both outputs 

# In[10]:


print('Original Text:')
display(HTML(f"<p style='font-size:15px; font-family:\"Comic Sans MS\", cursive;'> {original_text}</p>"))
print('\n')

print('Without Noise:')
display(HTML(f"<p style='font-size:15px; font-family:\"Comic Sans MS\", cursive;'>{baseline_text}</p>"))
print('\n')

print(f'With Noise (SNR = {target_snr} dB):')
display(HTML(f"<p style='font-size:15px; font-family:\"Comic Sans MS\", cursive;'>{noisy_text}</p>"))
print('\n')


# ## You can also display some of the variables you find interesting

# In[11]:


# `input_text` after tokenization
print(input_ids)

# its shape
print(input_ids.shape)


# In[12]:


# latent representation (context vectors)
print(encoder_outputs)

# its shape
print(encoder_outputs.last_hidden_state.shape)


# In[13]:


# The output of decoder before tokenization
print(baseline_outputs)

# its shape
print(baseline_outputs.shape)


# ## How `tokenizer` works
# Since we will be likely use the `Falconsai/text_summarization` model as our text summarizer. It is better to explore the tokenizer for T5 model

# In[14]:


# load tokenizer for T5
from transformers import AutoTokenizer
import torch

tokenizer_Fal = AutoTokenizer.from_pretrained("Falconsai/text_summarization")


# In[15]:


text = "I am a Master's student in Information and Networking Engineering at KTH."

# Tokenize the above text
tokenized_text = tokenizer_Fal(text, return_tensors="pt")

# By decoding the tokenized text you reconstruct the original text
decoded_text = tokenizer_Fal.decode(tokenized_text['input_ids'][0], skip_special_tokens=True)


# In[16]:


print(text)
print(tokenized_text['input_ids'][0])
print(decoded_text)


# In[17]:


# Changing the values of tokenized_text with change the decoded text
tokenized_text['input_ids'][0][0] = 6
tokenized_text['input_ids'][0][1] = 66
tokenized_text['input_ids'][0][-1] = 666
decoded_text_1 = tokenizer_Fal.decode(tokenized_text['input_ids'][0], skip_special_tokens=True)
print(decoded_text_1)

