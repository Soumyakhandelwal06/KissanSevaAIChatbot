---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:178816
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: asking about package of practice of cabbage
  sentences:
  - given detail information
  - 'spray 18 kg urea, 27 kg ssp and 6 kg mop per bigha


    '
  - recommended to spray rogor @ 2 ml per lit of water
- source_sentence: asking about treatment of mastitis in cow.
  sentences:
  - application of aminocel @ 2.5 ml per litre of water
  - '--advised to use ranjit sub -1 '
  - suggested to apply mastilep cream over the udder surface twice daily.
- source_sentence: query regarding management of nutrient in mustard.
  sentences:
  - 'told him to flush out the uterous with c-flox solution / vodine sol ution, inject
    enrocin 15 ml - 2 ml intra muscularly for3 days . inject dexona 1 ml -0.2 ml intra
    mascularly with the enrocin, eramix mixture - 5 g orally daily inject tonophosphen
    10 ml - 2 ml intra mascularly for 2 days, inject lutalyse - intra mascularly  &
    repeat the same  if necessary after 9-112 days '
  - advise to apply urea:ssp:mop @ 12:30:3 kg/bigha.
  - explained in detail
- source_sentence: asking that his 1-month old 2 goats are suffering from dirrhea.
  sentences:
  - either details are not registered in the portal or rejected due to wrong details.
  - recommended to spray classic 20 @ 2 ml per litr of water
  - 'tr.: sulcoprim.'
- source_sentence: asking about micronuitrient for rice
  sentences:
  - suggested to administer neblon powder @ 50 gm twice daily for 5 days for the treatment
    of diarrhoea iin cow.
  - explain in details
  - suggested to spray multiplex rice special @ 2gm/litre of water
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'asking about micronuitrient for rice',
    'suggested to spray multiplex rice special @ 2gm/litre of water',
    'suggested to administer neblon powder @ 50 gm twice daily for 5 days for the treatment of diarrhoea iin cow.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 1.0000, 1.0000],
#         [1.0000, 1.0000, 1.0000],
#         [1.0000, 1.0000, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 178,816 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 3 tokens</li><li>mean: 11.26 tokens</li><li>max: 131 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 23.29 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                      | sentence_1                                                             | label            |
  |:----------------------------------------------------------------|:-----------------------------------------------------------------------|:-----------------|
  | <code>asking about the to know kisan call centre</code>         | <code>explain in details</code>                                        | <code>1.0</code> |
  | <code>asking about the control of fruitfly in ridgegourd</code> | <code>suggest him to apply malathion 50 ec @ 2ml/litre of water</code> | <code>1.0</code> |
  | <code>horticulture related problem.</code>                      | <code>transfer to horticulture expert.</code>                          | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 4
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0895 | 500   | 0.0456        |
| 0.1790 | 1000  | 0.0001        |
| 0.2684 | 1500  | 0.0001        |
| 0.3579 | 2000  | 0.0001        |
| 0.4474 | 2500  | 0.0001        |
| 0.5369 | 3000  | 0.0           |
| 0.6263 | 3500  | 0.0           |
| 0.7158 | 4000  | 0.0           |
| 0.8053 | 4500  | 0.0           |
| 0.8948 | 5000  | 0.0           |
| 0.9843 | 5500  | 0.0           |
| 1.0737 | 6000  | 0.0           |
| 1.1632 | 6500  | 0.0           |
| 1.2527 | 7000  | 0.0           |
| 1.3422 | 7500  | 0.0           |
| 1.4316 | 8000  | 0.0           |
| 1.5211 | 8500  | 0.0           |
| 1.6106 | 9000  | 0.0           |
| 1.7001 | 9500  | 0.0           |
| 1.7895 | 10000 | 0.0           |
| 1.8790 | 10500 | 0.0           |
| 1.9685 | 11000 | 0.0           |
| 2.0580 | 11500 | 0.0           |
| 2.1475 | 12000 | 0.0           |
| 2.2369 | 12500 | 0.0           |
| 2.3264 | 13000 | 0.0           |
| 2.4159 | 13500 | 0.0           |
| 2.5054 | 14000 | 0.0           |
| 2.5948 | 14500 | 0.0           |
| 2.6843 | 15000 | 0.0           |
| 2.7738 | 15500 | 0.0           |
| 2.8633 | 16000 | 0.0           |
| 2.9528 | 16500 | 0.0           |
| 3.0422 | 17000 | 0.0           |
| 3.1317 | 17500 | 0.0           |
| 3.2212 | 18000 | 0.0           |
| 3.3107 | 18500 | 0.0           |
| 3.4001 | 19000 | 0.0           |
| 3.4896 | 19500 | 0.0           |
| 3.5791 | 20000 | 0.0           |
| 3.6686 | 20500 | 0.0           |
| 3.7581 | 21000 | 0.0           |
| 3.8475 | 21500 | 0.0           |
| 3.9370 | 22000 | 0.0           |


### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.1.2
- Transformers: 4.57.1
- PyTorch: 2.8.0+cu126
- Accelerate: 1.11.0
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->