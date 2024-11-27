import torch
import torch.nn.functional as F
import random
import numpy as np
import entropy_estimators as ee


# def add_noise_with_snr(encoder_output, target_snr_db):
#     """
#     Add noise to the encoder output based on a target SNR in dB.
    
#     Parameters:
#     - encoder_output: torch.Tensor, the encoder's output (last_hidden_state).
#     - target_snr_db: float, the desired signal-to-noise ratio in dB.
    
#     Returns:
#     - noisy_encoder_output: torch.Tensor, encoder output with added noise.
#     """
#     # Convert SNR from dB to linear scale
#     target_snr_linear = 10 ** (target_snr_db / 10)
    
#     # Calculate power of the signal
#     signal_power = torch.mean(encoder_output ** 2)
    
#     # Calculate required noise power for the target SNR
#     noise_power = signal_power / target_snr_linear
#     noise = torch.randn_like(encoder_output) * torch.sqrt(noise_power)
    
#     # Add noise to the encoder output
#     noisy_encoder_output = encoder_output + noise
#     return noisy_encoder_output

def add_noise_with_snr(encoder_output, noise_type='gaussian', target_snr_db=3, dropout_rate=0.4, sp_thresh=0.4):
    """
    Add noise to the encoder output based on a target SNR in dB.
    
    Parameters:
    - encoder_output: torch.Tensor, the encoder's output (last_hidden_state).
    - noise_type: string, determines what kind of noise is added.
    - target_snr_db: float, the desired signal-to-noise ratio in dB for awgn and dropout.
    - dropout_rate: float, range: [0,1], default rate for dropout noise (not used here).
    - sp_thresh: float, range: [0,1], determines the threshold for salt-and-pepper noise.
    
    Returns:
    - noisy_encoder_output: torch.Tensor, encoder output with added noise.
    """
    signal_power = torch.mean(encoder_output ** 2)
    target_snr = 10 ** (target_snr_db / 10)
    noise_power = signal_power / target_snr

    if noise_type.lower() == 'gaussian':
        # Generate Gaussian noise
        noise = torch.randn_like(encoder_output) * torch.sqrt(noise_power)
        return encoder_output + noise

    elif noise_type.lower() == 'dropout':
        random_tensor = torch.rand_like(encoder_output)
        mask = random_tensor >= dropout_rate
        noisy_encoder_output = encoder_output * mask.float()
        return noisy_encoder_output

    elif noise_type.lower() == 'saltpepper':
        mask = torch.rand_like(encoder_output) < sp_thresh  # The greater the sp_thresh, more noise is added
        salt = torch.max(encoder_output)
        pepper = torch.min(encoder_output)
        noise = torch.where(torch.rand_like(encoder_output) < 0.5, salt, pepper)
        noised_enc_output = torch.where(mask, noise, encoder_output)
        return noised_enc_output

    else:
        raise ValueError("Unsupported Noise Type. Choose between 'gaussian', 'dropout', 'saltpepper'.")
    

def masking(input_text, prob):
    '''
    Substitutes word with <mask> according to Bernoull-distribution.
    
    Parameters:
    - input_text: text to be masked partly
    - prob: probability that each word is replaced by a mask
    
    Returns:
    - masked_text: masked version of input_text
    '''
    flag = True
    while(flag):
        flag = False
        words = input_text.split()  # Split text into words
        masked_words = ["<mask>" if random.random() < prob else word for word in words]
        masked_text = " ".join(masked_words)
        
        # filter texts that have <mask> <mask> or <mask>. <mask>
        if "<mask> <mask>" in masked_text:
            flag = True
        if "<mask>. <mask>" in masked_text:
            flag = True
        
    return masked_text

def extract_hidden_states(decoder_hidden_states: tuple):
    """
    Extract the hidden states of last layer in decoder block
    """
    last_column = [row[-1] for row in decoder_hidden_states]
    last_layer_hs = torch.stack(last_column).squeeze()  

    return last_layer_hs

def align_tensors(tensor_a, tensor_b):
    """
    Aligns two tensors along the first dimension by padding.

    Args:
        tensor_a (torch.Tensor): The first tensor.
        tensor_b (torch.Tensor): The second tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The two tensors aligned along the first dimension.
    """
    # Get the first dimension sizes
    m, n = tensor_a.size(0), tensor_b.size(0)
    # Pad to the maximum length along the first dimension
    max_rows = max(m, n)
    if m < max_rows:
        padding = torch.zeros((max_rows - m, *tensor_a.shape[1:]), device=tensor_a.device)
        tensor_a = torch.cat([tensor_a, padding], dim=0)
    if n < max_rows:
        padding = torch.zeros((max_rows - n, *tensor_b.shape[1:]), device=tensor_b.device)
        tensor_b = torch.cat([tensor_b, padding], dim=0)

    return tensor_a, tensor_b

# This function is taken from the github of mutual information paper
def ksg(x, y, k=3):
    """
    Kraskov–Stogbauer–Grassberger (KSG) estimator of mutual information between two sentences represented as word embedding matrices x and y
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: KSG similarity measure between the two sentences
    """

    return ee.mi(x.T, y.T,k=3, base=np.e)

