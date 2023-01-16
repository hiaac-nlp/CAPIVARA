import json
from datasets import load_dataset
import pandas as pd

importCoco = 0
importConveptualCaptions = 0
WIT = 1

#Coco Train2017 Labels location and list
if importCoco:
    f = open('../../../datasets/diego.moreira/Coco/annotations/captions_train2017.json')
    dataset_train = json.load(f)
    f.close()
    
    # data train exemple
    for value in dataset_train['annotations']:
        print(value)
        #Remove Breack to list all labels
        break

    f = open('../../../datasets/diego.moreira/Coco/annotations/captions_val2017.json')
    dataset_valid = json.load(f)
    f.close()
    
    # data value exemple
    for value in dataset_valid['annotations']:
        print(value)
        #Remove Breack to list all labels
        break

#Conceptual Captions Labels location and list
if importConveptualCaptions:
    dataset = load_dataset('conceptual_captions')
    
    #data train exemple
    dataset['train'][0]

    
if WIT:
    wit_01 = pd.read_csv('../../../datasets/diego.moreira/WIT/wit_v1.train.all-00000-of-00010.tsv', sep='\t')
    # print(wit_01[wit_01.language.isin(['br', 'pt'])][['language','caption_reference_description']].head(50))
    print('br labels size : ' + str(wit_01[wit_01.language.isin(['br'])][['language','caption_reference_description']]))
    print('pt labels size : ' + str(wit_01[wit_01.language.isin(['pt'])][['language','caption_reference_description']]))

    
    
      # print(wit_01['language'].unique())
    # ['en' 'cs' 'sq' 'nl' 'de' 'pt' 'fa' 'es' 'be-tarask' 'ca' 'pl' 'ru' 'ur'
    #  'be' 'arz' 'ja' 'uk' 'hu' 'bg' 'zh-TW' 'az' 'fr' 'no' 'gl' 'hi' 'zh'
    #  'yue' 'sv' 'qu' 'ro' 'sl' 'sr' 'kk' 'lv' 'vi' 'ar' 'da' 'ms' 'it' 'af'
    #  'bn' nan 'nn' 'lt' 'sr-Latn' 'hr' 'ceb' 'et' 'eo' 'id' 'iw' 'hy' 'tr'
    #  'sco' 'fi' 'sk' 'el' 'hsb' 'mk' 'ast' 'nds' 'mg' 'jv' 'oc' 'br' 'ta' 'ko'
    #  'azb' 'eu' 'vec' 'la' 'cv' 'lah' 'ka' 'te' 'tt' 'ba' 'war' 'pa' 'is' 'tg'
    #  'th' 'ckb' 'fy' 'cy' 'ml' 'an' 'xmf' 'bs' 'mr' 'nv' 'ia' 'mn' 'si' 'lmo'
    #  'ga' 'sw' 'ne' 'uz' 'fil' 'ce' 'ht' 'my' 'kn' 'lb' 'vo' 'bar' 'io']