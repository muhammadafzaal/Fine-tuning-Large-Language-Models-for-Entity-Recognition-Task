# Fine-tuning-Large-Language-Models-for-Entity-Recognition-Task

## Task

1. Familiarize yourself with the MultiNERD Named Entity Recognition dataset (https://huggingface.co/datasets/Babelscape/multinerd)
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
1. Selected "bert-base-uncased" lanaguge model for this task.
2. Filtered English train, test, and validation files from the entite datasets
3. Fine-tunned bert-base-uncased twice on train and validation datasets \
   3.1. Model A: used all entity types as classes \
   3.2. Model B: used five entity types (PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)) as classes
4. Compared performance of both models on test dataset using accuracy, precision, recall and F1 metrics

## Results 
### Model A results 

Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1-score
--- | --- | --- | --- |--- |--- |--- 
1	| 0.076200	| 0.084466	| 0.969907	| 0.854900	| 0.832527	| 0.843565
2	| 0.056100	| 0.076390	| 0.972869	| 0.862263	| 0.871117	| 0.866668
3	| 0.041900	| 0.083289	| 0.973588	| 0.871504	| 0.872910	| 0.872206
4	| 0.029300	| 0.094933	| 0.973754	| 0.867077	| 0.883819	| 0.875368
5	| 0.021900	| 0.092688	| 0.974043	| 0.870094	| 0.883306	| 0.876650
6	| 0.013900	| 0.111109	| 0.974914	| 0.877645	| 0.886815	| 0.882206
7	| 0.009000	| 0.122158	| 0.973699	| 0.858838	| 0.900822	| 0.879329
8	| 0.005500	| 0.132115	| 0.974924	| 0.865756	| 0.902358	| 0.883678
9	| 0.003000	| 0.146467	| 0.975097	| 0.870216	| 0.899029	| 0.884388
10	| 0.001200	| 0.159002	| 0.975837	| 0.878465	| 0.898261	| 0.888253
--- | --- | --- | --- |--- |--- |--- 



