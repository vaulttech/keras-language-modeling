# Dataset wordnet_random_antonyms
# 1 layer
set -x

python insurance_qa_eval.py --model_name=MLPModel_100 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=MLPModel_200 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=MLPModel_300 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=MLPModel_400 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=MLPModel_500 --dataset_name=wordnet_random_antonyms


# Dataset wordnet_random_antonyms
# 2 layer

python insurance_qa_eval.py --model_name=DoubleMLPModel_100 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_200 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_300 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_400 --dataset_name=wordnet_random_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_500 --dataset_name=wordnet_random_antonyms


# Dataset wordnet_random_antonyms_deduplicate
# 1 layer

python insurance_qa_eval.py --model_name=MLPModel_100 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=MLPModel_200 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=MLPModel_300 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=MLPModel_400 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=MLPModel_500 --dataset_name=wordnet_random_antonyms_deduplicate


# Dataset wordnet_random_antonyms_deduplicate
# 2 layer

python insurance_qa_eval.py --model_name=DoubleMLPModel_100 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=DoubleMLPModel_200 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=DoubleMLPModel_300 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=DoubleMLPModel_400 --dataset_name=wordnet_random_antonyms_deduplicate
python insurance_qa_eval.py --model_name=DoubleMLPModel_500 --dataset_name=wordnet_random_antonyms_deduplicate


# Dataset wordnet_true_antonyms
# 1 layer

python insurance_qa_eval.py --model_name=MLPModel_100 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=MLPModel_200 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=MLPModel_300 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=MLPModel_400 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=MLPModel_500 --dataset_name=wordnet_true_antonyms


# Dataset wordnet_true_antonyms
# 2 layer

python insurance_qa_eval.py --model_name=DoubleMLPModel_100 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_200 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_300 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_400 --dataset_name=wordnet_true_antonyms
python insurance_qa_eval.py --model_name=DoubleMLPModel_500 --dataset_name=wordnet_true_antonyms


