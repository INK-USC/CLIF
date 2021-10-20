from .kilt_datasets import *
from .lamol_datasets import *
from .glue_datasets import *
from .leopard_datasets import *
from .mbpa_datasets import *
from .crossfit_qa_datasets import *
from .crossfit_cls_datasets import *
import random

KILT_TASKS=['fever','trex','structured_zeroshot','hotpotqa','nq','eli5','wow','aidayago2','triviaqa']
LAMOL_TASKS=['sst','woz.en','srl']
GLUE_TASKS = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'wnli','rte']
LEOPARD_TASKS = ['airline','conll','disaster','emotion','political_audience','political_bias','political_message','rating_books',
                 'rating_dvd','rating_electronics','rating_kitchen','scitail','sentiment_books','sentiment_dvd',
                 'sentiment_electronics','sentiment_kitchen','restaurant']
MBPA_TASKS = ['mbpa','mbpa:0','mbpa:1','mbpa:2','mbpa:3','mbpa:4']
CROSSFIT_QA_TRAIN_TASKS = [
    "ai2_arc",
    "aqua_rat",
    "boolq",
    "codah",
    "commonsense_qa",
    "cosmos_qa",
    "dream",
    "eli5-askh",
    "eli5-asks",
    "eli5-eli5",
    "freebase_qa",
    "hellaswag",
    "jeopardy",
    "kilt_hotpotqa",
    "kilt_nq",
    "kilt_trex",
    "kilt_zsre",
    "lama-conceptnet",
    "lama-google_re",
    "lama-squad",
    "lama-trex",
    "math_qa",
    "mc_taco",
    "numer_sense",
    "openbookqa",
    "qasc",
    "quail",
    "quarel",
    "quartz-no_knowledge",
    "quartz-with_knowledge",
    "race-high",
    "race-middle",
    "sciq",
    "search_qa",
    "social_i_qa",
    "squad-no_context",
    "superglue-copa",
    "superglue-multirc",
    "swag",
    "web_questions",
    "wino_grande",
    "wiqa"
]

CROSSFIT_QA_TEST_TASKS = [
    "adversarialqa",
    "biomrc",
    "duorc",
    "hotpot_qa",
    "quoref",
    "ropes",
    "squad-with_context",
    "superglue-record",
    "tweet_qa"
]

CROSSFIT_QA_METRICS = {
    'acronym_identification': 'EM',
    'ade_corpus_v2-classification': 'Classification-F1',
    'ade_corpus_v2-dosage': 'EM',
    'ade_corpus_v2-effect': 'EM',
    'adversarialqa': 'QA-F1',
    'aeslc': 'Rouge-L',
    'ag_news': 'Classification-F1',
    'ai2_arc': 'ACC',
    'amazon_polarity': 'Classification-F1',
    'anli': 'Classification-F1',
    'app_reviews': 'Pearson-Correlation',
    'aqua_rat': 'ACC',
    'art': 'ACC',
    'aslg_pc12': 'EM',
    'biomrc': 'QA-F1',
    'blimp-anaphor_gender_agreement': 'ACC',
    'blimp-anaphor_number_agreement': 'ACC',
    'blimp-determiner_noun_agreement_with_adj_irregular_1': 'ACC',
    'blimp-ellipsis_n_bar_1': 'ACC',
    'blimp-ellipsis_n_bar_2': 'ACC',
    'blimp-existential_there_quantifiers_1': 'ACC',
    'blimp-irregular_past_participle_adjectives': 'ACC',
    'blimp-sentential_negation_npi_licensor_present': 'ACC',
    'blimp-sentential_negation_npi_scope': 'ACC',
    'blimp-wh_questions_object_gap': 'ACC',
    'boolq': 'ACC',
    'break-QDMR': 'EM',
    'break-QDMR-high-level': 'EM',
    'circa': 'Classification-F1',
    'climate_fever': 'Classification-F1',
    'codah': 'Classification-F1',
    'common_gen': 'Rouge-L',
    'commonsense_qa': 'ACC',
    'cos_e': 'Rouge-L',
    'cosmos_qa': 'ACC',
    'crawl_domain': 'EM',
    'crows_pairs': 'ACC',
    'dbpedia_14': 'Classification-F1',
    'definite_pronoun_resolution': 'ACC',
    'discovery': 'Classification-F1',
    'dream': 'ACC',
    'duorc': 'QA-F1',
    'e2e_nlg_cleaned': 'Rouge-L',
    'eli5-askh': 'Rouge-L',
    'eli5-asks': 'Rouge-L', # dev
    'eli5-eli5': 'Rouge-L',
    'emo': 'Classification-F1',
    'emotion': 'Classification-F1',
    'empathetic_dialogues': 'Rouge-L',
    'ethos-directed_vs_generalized': 'Classification-F1',
    'ethos-disability': 'Classification-F1',
    'ethos-gender': 'Classification-F1',
    'ethos-national_origin': 'Classification-F1',
    'ethos-race': 'Classification-F1',
    'ethos-religion': 'Classification-F1',
    'ethos-sexual_orientation': 'Classification-F1',
    'financial_phrasebank': 'Classification-F1',
    'freebase_qa': 'EM',
    'gigaword': 'Rouge-L',
    'glue-cola': 'Matthew-Correlation',
    'glue-mnli': 'ACC',
    'glue-mrpc': 'ACC',
    'glue-qnli': 'ACC',
    'glue-qqp': 'ACC',
    'glue-rte': 'ACC',
    'glue-sst2': 'ACC',
    'glue-wnli': 'ACC',
    'google_wellformed_query': 'ACC',
    'hate_speech18': 'Classification-F1',
    'hate_speech_offensive': 'Classification-F1',
    'hatexplain': 'Classification-F1',
    'health_fact': 'Classification-F1',
    'hellaswag': 'ACC',
    'hotpot_qa': 'QA-F1',
    'imdb': 'Classification-F1',
    'jeopardy': 'EM',
    'kilt_ay2': 'EM',
    'kilt_fever': 'ACC',
    'kilt_hotpotqa': 'EM',
    'kilt_nq': 'EM',
    'kilt_trex': 'EM',
    'kilt_wow': 'Rouge-L',
    'kilt_zsre': 'EM',
    'lama-conceptnet': 'EM',
    'lama-google_re': 'EM',
    'lama-squad': 'EM',
    'lama-trex': 'EM',
    'liar': 'Classification-F1',
    'limit': 'EM',
    'math_qa': 'ACC',
    'mc_taco': 'ACC',
    'medical_questions_pairs': 'ACC',
    'mocha': 'Pearson-Correlation',
    'multi_news': 'Rouge-L',
    'numer_sense': 'EM',
    'onestop_english': 'Classification-F1',
    'openbookqa': 'ACC',
    'paws': 'Classification-F1',
    'piqa': 'ACC',
    'poem_sentiment': 'Classification-F1',
    'proto_qa': 'EM', # here
    'qa_srl': 'EM',
    'qasc': 'ACC',
    'quail': 'ACC',
    'quarel': 'ACC',
    'quartz-no_knowledge': 'ACC',
    'quartz-with_knowledge': 'ACC',
    'quoref': 'QA-F1',
    'race-high': 'ACC',
    'race-middle': 'ACC',
    'reddit_tifu-title': 'Rouge-L',
    'reddit_tifu-tldr': 'Rouge-L',
    'ropes': 'QA-F1',
    'rotten_tomatoes': 'Classification-F1',
    'samsum': 'Rouge-L',
    'scicite': 'Classification-F1',
    'sciq': 'ACC',
    'scitail': 'Classification-F1',
    'search_qa': 'EM',
    'sick': 'Classification-F1',
    'sms_spam': 'Classification-F1',
    'social_i_qa': 'ACC',
    'spider': 'EM',
    'squad-with_context': 'QA-F1',
    'squad-no_context': 'EM',
    'superglue-cb': 'ACC',
    'superglue-copa': 'ACC',
    'superglue-multirc': 'EM',
    'superglue-record': 'QA-F1',
    'superglue-rte': 'ACC',
    'superglue-wic': 'ACC',
    'superglue-wsc': 'ACC',
    'swag': 'ACC',
    'tab_fact': 'Classification-F1',
    'trec': 'Classification-F1',
    'trec-finegrained': 'Classification-F1',
    'tweet_eval-emoji': 'Classification-F1',
    'tweet_eval-emotion': 'Classification-F1',
    'tweet_eval-hate': 'Classification-F1',
    'tweet_eval-irony': 'Classification-F1',
    'tweet_eval-offensive': 'Classification-F1',
    'tweet_eval-sentiment': 'Classification-F1',
    'tweet_eval-stance_abortion': 'Classification-F1',
    'tweet_eval-stance_atheism': 'Classification-F1',
    'tweet_eval-stance_climate': 'Classification-F1',
    'tweet_eval-stance_feminist': 'Classification-F1',
    'tweet_eval-stance_hillary': 'Classification-F1',
    'tweet_qa': 'QA-F1',
    'web_questions': 'EM',
    'wiki_auto': 'Classification-F1',
    'wiki_bio': 'Rouge-L',
    'wiki_qa': 'Classification-F1',
    'wiki_split': 'Rouge-L',
    'wikisql': 'EM',
    'wino_grande': 'ACC',
    'wiqa': 'ACC',
    'xsum': 'Rouge-L',
    'yahoo_answers_topics': 'Classification-F1',
    'yelp_polarity': 'Classification-F1',
    'yelp_review_full': 'Pearson-Correlation'
}

CROSSFIT_CLS_TRAIN_TASKS = [
        "superglue-rte", "tweet_eval-sentiment", "discovery", "glue-rte", "superglue-wsc", "scicite",
        "glue-mrpc", "tweet_eval-stance_hillary", "tweet_eval-offensive", "emotion", "hatexplain", "glue-cola", "sick",
        "paws", "ethos-sexual_orientation", "glue-qqp", "tweet_eval-emotion", "sms_spam",
        "health_fact", "glue-mnli", "imdb", "ethos-disability", "glue-wnli", "scitail", "trec-finegrained",
        "yahoo_answers_topics", "liar", "glue-sst2", "tweet_eval-stance_abortion", "circa", "tweet_eval-stance_climate",
        "glue-qnli", "tweet_eval-emoji", "ethos-directed_vs_generalized", "ade_corpus_v2-classification",
        "wiki_auto", "hate_speech_offensive", "superglue-wic", "google_wellformed_query",
        "tweet_eval-irony", "ethos-gender", "onestop_english", "trec", "rotten_tomatoes", "kilt_fever"
    ]
CROSSFIT_CLS_TEST_TASKS = [
"superglue-cb", "dbpedia_14", "wiki_qa", "emo", "yelp_polarity", "ethos-religion", "financial_phrasebank", "tab_fact", "anli", "ethos-race"
]

def task_collection_to_tasks(collection_full_name):
    items = collection_full_name.split(':')
    collection_name = items[0]

    tasks = None
    if collection_name == 'kilt':
        tasks = KILT_TASKS
    elif collection_name == 'lamol':
        tasks = LAMOL_TASKS
    elif collection_name == 'glue':
        tasks = GLUE_TASKS
    elif collection_name == 'leopard':
        tasks = LEOPARD_TASKS
    elif collection_name == 'mbpa':
        tasks = MBPA_TASKS
    elif collection_name == 'crossfit_qa_train':
        tasks = CROSSFIT_QA_TRAIN_TASKS
    elif collection_name == 'crossfit_qa_test':
        tasks = CROSSFIT_QA_TEST_TASKS
    elif collection_name == 'crossfit_cls_train':
        tasks = CROSSFIT_CLS_TRAIN_TASKS
    elif collection_name == 'crossfit_cls_test':
        tasks = CROSSFIT_CLS_TEST_TASKS

    if len(items) > 1:
        start = int(items[1])
        stop = int(items[2])
        tasks = tasks[start:stop]

    return tasks


def get_main_metrics(task_name):
    if task_name in CROSSFIT_QA_METRICS:
        met = CROSSFIT_QA_METRICS[task_name]
        if met in ['EM','ACC']:
            return 'em'
        else:
            return 'f1'
    return 'em'

def get_dataset(args, task_name, split, tokenizer: PreTrainedTokenizer,
                 gen_token=None, full_init=True, **kwargs):
    if task_name in KILT_TASKS:
        DATASET_CLS = KILTDatasetWOContext
    elif task_name in LAMOL_TASKS:
        DATASET_CLS = LAMOLDataset
    elif task_name in GLUE_TASKS:
        DATASET_CLS = GLUEDataset
    elif task_name in LEOPARD_TASKS and (not args.task_collection or args.task_collection == 'leopard'):
        DATASET_CLS = LeopardDataset
    elif task_name in MBPA_TASKS:
        DATASET_CLS = MBPADataset
    elif task_name in CROSSFIT_QA_TEST_TASKS or task_name in CROSSFIT_QA_TRAIN_TASKS and \
            (not args.task_collection or args.task_collection in ['crossfit_qa_train','crossfit_qa_test']):
        DATASET_CLS = CrossFitQADataset
    elif task_name in CROSSFIT_CLS_TEST_TASKS or task_name in CROSSFIT_CLS_TRAIN_TASKS and \
            (not args.task_collection or args.task_collection in ['crossfit_cls_train', 'crossfit_cls_test']):
        DATASET_CLS = CrossFitCLSDataset
    else:
        raise NotImplementedError
    dataset = DATASET_CLS(args, task_name, split, tokenizer, gen_token, full_init=full_init, **kwargs)
    return dataset