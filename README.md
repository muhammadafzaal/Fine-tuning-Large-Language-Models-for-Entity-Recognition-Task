# Fine-tuning-Large-Language-Models-for-Entity-Recognition-Task

## Tasks 

1. Familiarize yourself with the MultiNERD Named Entity Recognition dataset
2. Find a suitable LM model on HuggingFace Model Hub (https://huggingface.co/models). This can
be a Large Language Model (LLM) or any type of Transformer-based Language Model
3. Filter out the non-English examples of the dataset
4. Fine-tune your chosen model on the English subset of the training set. This will be system A
5. You will now train a model that will predict only five entity types and the O tag (I.e. not part of
an entity). Therefore, you should perform the necessary pre-processing steps on the dataset. All
examples should thus remain, but entity types not belonging to one of the following five should
be set to zero: PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS),
ANIMAL(ANIM)
6. Fine-tune your model on the filtered dataset that you constructed in step 5. This will be system
B
7. Pick a suitable metric or suite of metrics and evaluate both systems A and B using the test set

## Implemetation  
1. Selected "bert-base-uncased" lanaguge model for this tasks.
2. Filtered English train, test, and validation datasets from the entite datasets
3. Fine-tunned two bert-base-uncased two times
   3.1. Model A: used all entity types
   3.2. Model B: used five entity types (PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM))
