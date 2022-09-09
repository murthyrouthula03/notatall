Files and Datasets paths:
------------------------

Python files: Codes\trails\

Kabita's dataset:
----------------

Input dataset, stopwords list, pre-processing data and final test data : Codes\trails\Datasets\Kabita\Input\

Bag-of-word vectorizers datasets : Codes\trails\Datasets\Kabita\BagOfWords\

Sentence Transformer datasets : Codes\trails\Datasets\Kabita\SentenceTransformers\

Fine-tuned Transformer datasets : Codes\trails\Datasets\Kabita\FineTunedTransformers\


Nisha's dataset:
----------------

Input dataset, stopwords list, pre-processing data and final test data : Codes\trails\Datasets\Nisha\Input\

Bag-of-word vectorizers datasets : Codes\trails\Datasets\Nisha\BagOfWords\

Sentence Transformer datasets : Codes\trails\Datasets\Nisha\SentenceTransformers\

Fine-tuned Transformer datasets : Codes\trails\Datasets\Nisha\FineTunedTransformers\

Saved models of Kabita and Nisha dataset:
----------------------------------------

Codes\trails\Final_Models\

=============================================================================================================

Coding is done in the following steps for both Kabita and Nisha datasets

1.	Data Understanding and Visualization
2.	Vectorization using Bag-of-word models
3.	Vectorization using pre-trained Transformers
4.	Modeling on vectorized datasets and Evaluation
5.	Hypothesis testing on the models
6.	Hyperparameter tuning
7.	Final model training, saving and prediction


steps - 1, 2, 3, 4, 5, and 6 are coded individually for Kabita and Nisha datasets.

step - 7 is coded in the same files for both Kabita and Nisha datasets.

=============================================================================================================

Python filenames and description:
--------------------------------

Kabita dataset:
--------------

1.) Data Understanding and Visualization:
    ------------------------------------

filename: Kabita_Dataset_DataVisualization.ipynb

file description: Visualizations for Kabita dataset are coded in this file.

2.) Vectorization using Bag-of-word models:
    --------------------------------------

filename: Kabita_Dataset_BagofWords_Vectorizations.ipynb

file description: Data Cleaning and Vectorizations of Kabita dataset using TF-IDF, Count, and Term Frequency are coded in this file.

3.) Vectorization using pre-trained Transformers:
    --------------------------------------------

Sentence Transformers:
---------------------

filename: Kabita_Dataset_Transformers_Vectorized.ipynb

file description: Vectorization of Kabita dataset using Sentence Transformer are coded in this file

Fine-tuned Transformers:
-----------------------

filename: Kabita_Dataset_finetuned_model_bert_base_vectors.ipynb

file description: Vectorization of Kabita dataset using Fine-tuned BERT Base Transformer are coded in this file.

*************************************************************************************************************

filename: Kabita_Dataset_finetuned_model_bert_hinglish_vectors.ipynb

file description: Vectorization of Kabita dataset using Fine-tuned BERT Hinglish Transformer are coded in this file.

*************************************************************************************************************

filename: Kabita_Dataset_finetuned_model_gpt_base_vectors.ipynb

file description: Vectorization of Kabita dataset using Fine-tuned GPT Base Transformer are coded in this file.

*************************************************************************************************************

filename: Kabita_Dataset_finetuned_model_gpt_hinglish_vectors.ipynb

file description: Vectorization of Kabita dataset using Fine-tuned GPT Hinglish Transformer are coded in this file.

*************************************************************************************************************

filename: Kabita_Dataset_finetuned_model_xlm_base_vectors.ipynb

file description: Vectorization of Kabita dataset using Fine-tuned XLM Base Transformer are coded in this file.


4.) Modeling on vectorized datasets and Evaluation:
    ----------------------------------------------

Models without scaling and Component Analysis:
---------------------------------------------

filename: Kabita_Dataset_BagOfWords_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on Bag-of-words models of Kabita's dataset without scaling and component analysis.

*************************************************************************************************************

filename: Kabita_Dataset_Transformers_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on Transformer models of Kabita's dataset without scaling and component analysis.

Models with Scaling:
-------------------

filename: Kabita_MinMaxScaling_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Kabita's dataset with Min-Max scaling.

*************************************************************************************************************

filename: Kabita_NormalizeScaling_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Kabita's dataset with Normalize scaling.

*************************************************************************************************************

filename: Kabita_StandardScaling_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Kabita's dataset with Standard scaling.

Models with Component Analysis:
------------------------------

filename: Kabita_PCA_Scree_Plots.ipynb

file description: Scree plots are plotted for Kabita's dataset for getting number of components for PCA using elbow method.

*************************************************************************************************************

filename: Kabita_PCA_ICA_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Kabita's dataset after component analysis based on scree plots.

5.) Hypothesis testing on the models:
    --------------------------------

filename: Kabita_Dataset_HypothesisTest.ipynb

file description: Hypothesis test is done on the efficient models of Kabita's dataset.

6.) Hyperparameter tuning:
    ---------------------

filename: Kabita_Dataset_HyperParameter_Tuning.ipynb

file description: Hyperparameter tuning is done on the efficient models of Kabita's dataset.

*************************************************************************************************************

filename: Kabita_Dataset_AUC_ROC.ipynb

file description: AUC-ROC curves are plotted for the efficient models of Kabita's dataset after Hyperparameter tuning.


Nisha dataset:
-------------

1.) Data Understanding and Visualization:
    ------------------------------------

filename: Nisha_Dataset_DataVisualization.ipynb

file description: Visualizations for Nisha dataset are coded in this file.

2.) Vectorization using Bag-of-word models:
    --------------------------------------

filename: Nisha_Dataset_BagofWords_Vectorizations.ipynb

file description: Data Cleaning and Vectorizations of Nisha dataset using TF-IDF, Count, and Term Frequency are coded in this file.

3.) Vectorization using pre-trained Transformers:
    --------------------------------------------

Sentence Transformers:
---------------------

filename: Nisha_Dataset_Transformers_Vectorized.ipynb

file description: Vectorization of Nisha dataset using Sentence Transformer are coded in this file

Fine-tuned Transformers:
-----------------------

filename: Nisha_Dataset_finetuned_model_bert_base_vectors.ipynb

file description: Vectorization of Nisha dataset using Fine-tuned BERT Base Transformer are coded in this file.

*************************************************************************************************************

filename: Nisha_Dataset_finetuned_model_bert_hinglish_vectors.ipynb

file description: Vectorization of Nisha dataset using Fine-tuned BERT Hinglish Transformer are coded in this file.

*************************************************************************************************************

filename: Nisha_Dataset_finetuned_model_gpt_base_vectors.ipynb

file description: Vectorization of Nisha dataset using Fine-tuned GPT Base Transformer are coded in this file.

*************************************************************************************************************

filename: Nisha_Dataset_finetuned_model_gpt_hinglish_vectors.ipynb

file description: Vectorization of Nisha dataset using Fine-tuned GPT Hinglish Transformer are coded in this file.

*************************************************************************************************************

filename: Nisha_Dataset_finetuned_model_xlm_base_vectors.ipynb

file description: Vectorization of Nisha dataset using Fine-tuned XLM Base Transformer are coded in this file.


4.) Modeling on vectorized datasets and Evaluation:
    ----------------------------------------------

Models without scaling and Component Analysis:
---------------------------------------------

filename: Nisha_Dataset_BagOfWords_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on Bag-of-words models of Nisha's dataset without scaling and component analysis.

*************************************************************************************************************

filename: Nisha_Dataset_Transformers_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on Transformer models of Nisha's dataset without scaling and component analysis.

Models with Scaling:
-------------------

filename: Nisha_MinMaxScaling_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Nisha's dataset with Min-Max scaling.

*************************************************************************************************************

filename: Nisha_NormalizeScaling_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Nisha's dataset with Normalize scaling.

*************************************************************************************************************

filename: Nisha_StandardScaling_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Nisha's dataset with Standard scaling.

Models with Component Analysis:
------------------------------

filename: Nisha_PCA_Scree_Plots.ipynb

file description: Scree plots are plotted for Nisha's dataset for getting number of components for PCA using elbow method.

*************************************************************************************************************

filename: Nisha_PCA_ICA_ML_Models.ipynb

file description: Machine Learning models are trained and evaluated on all vectorized models of Nisha's dataset after component analysis based on scree plots.

5.) Hypothesis testing on the models:
    --------------------------------

filename: Nisha_Dataset_HypothesisTest.ipynb

file description: Hypothesis test is done on the efficient models of Nisha's dataset.

6.) Hyperparameter tuning:
    ---------------------

filename: Nisha_Dataset_HyperParameter_Tuning.ipynb

file description: Hyperparameter tuning is done on the efficient models of Nisha's dataset.

*************************************************************************************************************

filename: Nisha_Dataset_AUC_ROC.ipynb

file description: AUC-ROC curves are plotted for the efficient models of Nisha's dataset after Hyperparameter tuning.

=============================================================================================================

7.) Final model training, saving and prediction:
    -------------------------------------------

filename: Final_Models_Saving_Pickle_File_Creation.ipynb

file description: Final models of both Kabita and Nisha datasets are trained and saved with this code.

*************************************************************************************************************

filename: Final_Models_Predictions.ipynb

file description: Random data is tested with the saved models of Kabita and Nisha datasets with this code.

=============================================================================================================




