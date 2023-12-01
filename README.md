# Fine-tuning-Large-Language-Models-for-Entity-Recognition-Task

## Instructions to Run
1. git clone https://github.com/muhammadafzaal/Fine-tuning-Large-Language-Models-for-Entity-Recognition-Task.git
2. pip install -r requirements.txt
3. python code.py

**IMPORTANT:** "filtered_categories = True" in code.py means only five entity types (PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM)) will be used as classes. 

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
### Model A 

#### Training 

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

#### Testing 
Testing Loss | Accuracy | Precision | Recall | F1-score
--- | --- |--- |--- |--- 
0.125017	| 0.981275	| 0.911197	| 0.927331	| 0.919193

### Model B 

#### Training 

Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1-score
--- | --- | --- | --- |--- |--- |--- 
1	| 0.050000	| 0.047890	| 0.983887	| 0.894167	| 0.915788	| 0.904848
2	| 0.035500	| 0.044516	| 0.985665	| 0.910056	| 0.925460	| 0.917694
3	| 0.024800	| 0.046571	| 0.985656	| 0.915898	| 0.916046	| 0.915972
4	| 0.016100	| 0.052598	| 0.985434	| 0.907333	| 0.923042	| 0.915120
5	| 0.012100	| 0.058889	| 0.985663	| 0.907635	| 0.926685	| 0.917061
6	| 0.008100	| 0.064475	| 0.985411	| 0.910599	| 0.924074	| 0.917287
7	| 0.004100	| 0.074972	| 0.985702	| 0.910561	| 0.927588	| 0.918996
8	| 0.002400	| 0.081268	| 0.985949	| 0.912273	| 0.929361	| 0.920738
9	| 0.001300	| 0.086472	| 0.986297	| 0.913485	| 0.931715	| 0.922510
10	| 0.000500	| 0.095753	| 0.986309	| 0.917662	| 0.928845	| 0.923220
--- | --- | --- | --- |--- |--- |--- 

#### Testing 
Testing Loss | Accuracy | Precision | Recall | F1-score
--- | --- |--- |--- |--- 
0.090996	| 0.987007	| 0.933750	| 0.945388	| 0.939533

## Conclusion   
In the context of model A, it is noteworthy to observe a substantial decrease in training loss from 0.076200 during the initial epoch to 0.001200 by the tenth epoch. This reduction in training loss suggests that the model has exhibited a commendable capacity for learning. The observed trend in the validation Loss indicates a notable increase over the course of multiple epochs, which may suggest the presence of potential overfitting.During all phases of the task, it was observed that remaining metrics exhibited improvement, particularly in the case of recall, which displayed a noteworthy increase from an initial value of 83.25% to a final value of 89.83%. In the context of Model B, it is observed that the training loss exhibited a notable decline over the course of the training process. Specifically, the training loss decreased from an initial value of 0.050000 in epoch 1 to a significantly lower value of 0.000500 by epoch 10. However, it is worth noting that the validation loss of Model B was observed to be lower than that of Model A. This suggests that Model B may have performed better in terms of generalisation, as a lower validation loss is typically associated with a model that is less prone to overfitting. The observed trend indicates a higher level of consistency in the improvement of remaining metrics, relative to Model A.

In order to mitigate the risk of overfitting during the fine-tuning process of BERT for entity recognition, it is advisable to incorporate regularisation techniques such as dropout, employ early stopping, and conduct cross-validation. These measures can help prevent the model from excessively fitting the training data and enhance its generalisation capabilities. In addition, it is helpful to diligently monitor the metrics related to overfitting and contemplate the utilisation of ensemble methods as a means to enhance the generalisation capabilities of the model.










