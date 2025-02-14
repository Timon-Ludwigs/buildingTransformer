# Transformer

This repository contains a PyTorch implementation of the Transformer model as originally introduced by Vaswani et al. in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)*. It is the result of my work in the Implementing Transformers course during the Winter Semester 2024/25 at the [Heinrich Heine University Düsseldorf](https://www.heicad.hhu.de/lehre/masters-programme-ai-and-data-science), led by [Carel van Niekerk](https://carelvniekerk.github.io/).

With numerous Transformer implementations available, what makes this one unique? In short, I successfully train and validate the model on an NVIDIA A100 using mixed precision (fp16), which comes with its own challenges. This repository shares the techniques and optimizations I used to make it work efficiently.

Additionally, you will find my written report on the code and course below. It includes intuitive explanations and mathematical insights that I discovered or derived during my research.

## Schedule

| Week | Dates         | Practical                                              |
|------|---------------|--------------------------------------------------------|
| 1    | 7-11.10.2024  | Practical 1: Getting Started and Introduction to Transformers and Attention |
| 2    | 14-18.10.2024 | Practical 2: Introduction to Unit Tests and Masked Attention |
| 3    | 21-25.10.2024 | Practical 3: Tokenization                              |
| 4    | 28-31.10.2024 | Practical 4: Data Preparation and Embedding Layers     |
| 5    | 4-8.11.2024   | Practical 5: Multi-Head Attention Blocks               |
| 6    | 11-15.11.2024 | Practical 6: Transformer Encoder and Decoder Layers    |        
| 7    | 18-22.11.2024 | Practical 6: Transformer Encoder and Decoder Layers    | 
| 8    | 25-29.11.2024 | Practical 7: Complete Transformer Model                | 
| 9    | 2-6.12.2024   | Practical 8: Training and Learning rate Schedules      |                                                
| 10   | 9-13.12.2024  | Practical 9: Training the model                        |
| 11   | 16-20.12.2024 | Practical 10: Training the model                       |                                               
| 12   | 6-11.01.2025  | Practical 11: Autoregressive Generation and Evaluation |                                                
| 13   | 13-17.01.2025 | Practical 12: GPU Training (HPC)                       |                                                
| 14   | 29.01.2025    | Deadline of written report                             |  
| 14   | 13.02.2025    | Oral presentation in person                            | 

## Report

**Report Guidelines:**
**Word Limit:** The report should not exceed 2500 words.
**Page Limit:** The report must be a maximum of 8 pages.

[Report](transformer_report.pdf)
