# Information-Secrecy-using-LLMs
### Sreyan Ghosh

Working on the noising of the encoder output which is to be fed into the `generate` method of the `BartConditionalModel`

Possible avenues to explore:
- [x] Simple AWGN addition (done by Ziyue)
- [ ] Explore the `Trainer` class' `NEFTune` method.
- [ ] Explre the `SFTTrainer` class which is Supervised Fine Tuning.
