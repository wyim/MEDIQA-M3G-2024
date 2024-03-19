# MEDIQA-M3G-2024

* Website: <https://sites.google.com/view/mediqa2024/mediqa-m3g>

* MEDIQA-M3G@Codabench: <https://www.codabench.org/competitions/1632/>

# Task Description

This dataset includes clinical dermatology textual queries and their associated images, as well as the answers to the queries.

The languages used here are Chinese (zh), English (en), and Spanish (es).


# Data Splits
The following table is the number of instances for each split.

|Train|Valid|Test|
| -------- | ------- |------- |
| 842|56|100|

- There will be a Chinese, English, and Spanish version of each. 
- All non-Chinese train splits are machine translated from the Chinese original. 
- All valid/test sets are human translated and validated by medical doctors.

# Data Description

Input Content

(a) json list where each instance will be represented by a json object with the following attributes:

| attribute_id | description |
| -------- | ------- |
|encounter_id|unique identification string for the case|
|image_ids|list of strings of the image_ids|
|query_title_{LANGUAGE}|a string representing the query title|
|query_content_{LANGUAGE}|a string representing the query content|

(b) image files with unique ids

Images are stored in the following folders respectively:
images_{train,valid,test}/

Image naming convention is as follows: IMG_{ENCOUNTER_ID}_{IMAGENUM}.*

(c) df_userinfo.csv

csv where each row will include author attributes:

|attribute_id|description|
| -------- | ------- |
|author_id|unique identification string for the author|
|validation_level|validation level of the response author obtained by uploading their professional credentials (categories are: {realid_validated, md_validated, md1_validated, md2_validated, md3_validated, md4_validated}) - md{} - indicates some medical doctor certificate uploaded, the higher the number, the higher the rank. (Note: no validation level doesn’t mean author is not a doctor)|
|rank_level|author leveling system based on past useful response ratings (level_{0-8}, higher the better)|

(d) Reference data will additionally have the field:

|attribute_id|description|
| -------- | ------- |
|responses|a list of json objects with the following keys ( author_id, content_{LANGUAGE} , completeness, contains_freq_ans)|

completeness: given a score {0.0, 0.5, 1.0}, depending on if the original queries’ question was fully answered. A score of 1.0 indicates that the query was completely answered, 0.5 if partially answered, 0.0 if not answered. If no explicit query is given, we assume the query asks for a disease AND treatment.

contains_freq_ans: given a score {0.0, 1.0}. 1.0 is given if the most frequent answer.


# Output Content

Output should be json list with at least the following content

|attribute_id|description|
| -------- | ------- |
|encounter_id|unique identification string for the case|
|responses|a list of json objects with the following keys ( “content_{LANGUAGE}” ) - put your answer in first object|

# Evaluation Script

**We use the following evaluation metrics:**
- **DeltaBLEU** and **bertscore** for general NLG evaluation. DeltaBLEU weighs n-grams according to a human-evaluated score. BERTSCORE takes the maximum score for any available reference. The evaluation scripts used are in evaluation/nlg.

- **MEDCON** for medical concept and assertion evaluation. QUICKUMLS is used to identify concepts (<https://github.com/Georgetown-IR-Lab/QuickUMLS>), assertion classification is obtained using an in-house Llama classifier for English and GPT-4 for Chinese. The assertion models are not released, please approximate using your own classifiers. Evaluation scripts are found in evaluation/medcon.


# Contact
 MEDIQA-NLP mailing list: https://groups.google.com/g/mediqa-nlp 
 Email: mediqa.organizers@gmail.com 

# Organizers   
* Asma Ben Abacha, Microsoft, USA
* Wen-wai Yim, Microsoft, USA
* Meliha Yetisgen, University of Washington, USA
* Fei Xia, University of Washington, USA
