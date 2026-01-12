# GenAI-Creative-Text-and-Image-Generation

Generative AI Project: Transformers & GANs

This repository contains my work on a comprehensive Generative AI project covering both text generation and image generation models. The project focuses on understanding and implementing modern generative architectures, including transformers, GPT-2 fine-tuning, and multiple GAN variants, through hands-on experimentation and implementation.

Project Overview

The project is divided into two major phases:

Phase 1: Text-based Generative AI using Transformers and GPT-style models

Phase 2: Image-based Generative AI using Generative Adversarial Networks (GANs)

The work emphasizes dataset preparation, model implementation, training, evaluation, and practical experimentation with generative systems.

Phase 1: Generative AI with Transformers
Dataset Preparation

Curated a creative text dataset (e.g., poetry, short stories, or recipes) from publicly available sources.

Cleaned and normalized raw text data.

Tokenized text using word-level and subword-level tokenization.

Prepared datasets suitable for transformer-based training.

Transformer Fundamentals

Studied attention mechanisms, including self-attention and multi-head attention.

Understood positional encoding and encoder–decoder architectures.

Compared transformers with RNN, LSTM, and GRU-based models.

Building a Transformer from Scratch

Implemented a basic transformer model using PyTorch.

Developed core components:

Self-attention mechanism

Positional encoding

Encoder and decoder layers

Trained the model for next-token prediction tasks.

Evaluated performance using cross-entropy loss and qualitative text generation.

Generated sample outputs to analyze contextual coherence.

Fine-Tuning GPT-2

Fine-tuned a pre-trained GPT-2 model on the curated creative dataset.

Used Hugging Face Transformers for training and inference.

Evaluated generated text using qualitative assessment and standard NLP metrics.

Experimented with different prompts to improve output quality.

Uploaded the fine-tuned model for reuse and inference.

Phase 2: Generative Adversarial Networks (GANs)
Introduction to GANs

Studied the generator–discriminator framework.

Implemented vanilla GANs using TensorFlow.

Understood binary cross-entropy loss and minimax optimization.

FCGAN and DCGAN

Implemented Fully Connected GANs (FCGAN) and Deep Convolutional GANs (DCGAN).

Compared training stability and output quality across architectures.

Wasserstein GANs

Implemented Wasserstein GANs (WGAN) to improve training stability.

Studied the benefits of Wasserstein loss over standard GAN loss functions.

Conditional GANs

Implemented Conditional GANs (cGANs) for controlled image generation.

Generated outputs conditioned on class labels or attributes.

Technologies Used

Python

PyTorch

TensorFlow

Hugging Face Transformers

NumPy, Pandas, Matplotlib

Outcomes

Hands-on experience with transformer architectures and attention mechanisms.

Practical understanding of fine-tuning large language models like GPT-2.

Experience implementing and comparing multiple GAN variants.

Strong foundation in generative model training, evaluation, and experimentation.
