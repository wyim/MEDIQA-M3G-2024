import json
import os
import numpy as np

import sys

#modified from https://github.com/mgalley/sacreBLEU
import sacrebleu_deltableu
from bert_score import score

try:
    import pandas as pd
except ImportError as e:
    print( e )
    pass  # module doesn't exist, deal with it.


def read_json( file_path ) :
    with open( file_path, 'r' ) as f :
        return json.load( f )


def get_weight_schemes( df_userinfo, author_id, completeness, contains_freq_ans ) :

    if df_userinfo is None :
        return 1.0

    userrow = df_userinfo[ df_userinfo['author_id']==author_id ].iloc[0]
    user_validlevel = userrow[ 'validation_level' ]
    user_rank = userrow[ 'rank_level' ]
    if user_rank == 'unk' :
        user_rank = 'level_0'

    answer_profile = [
           [ 1 if str(user_validlevel)[:2]=='md' else 0 ][0],
           [ 1 if int( user_rank.replace('level_','') ) >=4 else 0 ][0],
           [ 1 if contains_freq_ans==1.0 else 0 ][0],
           [ 1 if completeness==1.0 else 0 ][0],
        ]
    
    w1 = -100
        
    if answer_profile[-1] == 1 :
        first_3 = sum( answer_profile[:-1] )
        if first_3 == 3 :
            w1 = 1.0
        elif first_3 == 2 :
            w1 = 0.9
        elif first_3 == 1 :
            w1 = 0.8
        else :
            w1 = 0.7
    else :
        first_3 = sum( answer_profile[:-1] )
        if first_3 == 3 :
            w1 = 0.9
        elif first_3 == 2 :
            w1 = 0.8
        elif first_3 == 1 :
            w1 = 0.7
        else :
            w1 = 0.6

    return w1


def get_ref_scores( references, df_userinfo ) :

    ref_scores = []

    for reference in references :
        author_id = reference[ 'author_id' ]
        completeness = reference[ 'completeness' ]
        contains_freq_ans = reference[ 'contains_freq_ans' ]
        ref_scores.append( get_weight_schemes( df_userinfo, author_id, completeness, contains_freq_ans ) )

    return ref_scores


# #############

def main( truth, prediction, score_path, df_userinfo=None ) :

    print( 'Number of instances: {}'.format( len( truth ) ) )
    reference_langs = [ x.replace( 'content_', '' ) for x in truth[0]['responses'][0].keys() if 'content_' in x ]
    reference_ids = [ x['encounter_id'] for x in truth ]
    print( 'Reference languages: {}, {}'.format( len(reference_langs), str( reference_langs ) ) )

    print( 'Number of instances: {}'.format( len( prediction ) ) )
    prediction_langs = [ x.replace( 'content_', '' ) for x in prediction[0]['responses'][0].keys() if 'content_' in x ]
    prediction_ids = [ x['encounter_id'] for x in truth ]
    print( 'Predicted languages: {}, {}'.format( len(prediction_langs), str( prediction_langs ) ) )

    #checking that the encounter id's are the same and that instance id's
    print( 'Checking instance ids match.')
    bad_match = 0
    for ind, reference_id in enumerate( reference_ids ) :
        prediction_id = prediction_ids[ ind ]
        if reference_id != prediction_id :
            bad_match += 1
            print( 'INDEX {} has different ids for reference and prediction, {} and {} respectively.'.format( ind, reference_id, prediction_id ) )

    if bad_match > 0 :
        print('Please check that your encounter id for your prediction and input are in the same order!!')
        sys.exit(0)

    print('Calculating evaluation')

    prediction_langs = list( set( prediction_langs ) & set( reference_langs ) )

    references = {}
    predictions = {}
    reference_weights = []

    for prediction_lang in prediction_langs :
        references[ prediction_lang ] = []
        predictions[ prediction_lang ] = []

    for ind, reference_instance in enumerate( truth ) :

        # dealing with reddit, where the score is always 1
        if 'zh' in prediction_langs:
            reference_weights.append( get_ref_scores( reference_instance[ 'responses' ], df_userinfo ) )
        else:
            reference_weights.append([1]*len(reference_instance[ 'responses' ]))
        for prediction_lang in prediction_langs :
            
            refs = [ x[ 'content_{}'.format( prediction_lang ) ] for x in reference_instance[ 'responses' ] ]
            hyp = prediction[ ind ][ 'responses' ][0][ 'content_{}'.format( prediction_lang ) ]
            
            references[ prediction_lang ].append( refs )
            predictions[ prediction_lang ].append( hyp )

    print('Scores:')
    scores = {}

    for pred_lang in prediction_langs :

        if pred_lang == 'zh' :
            delatbleu = sacrebleu_deltableu.corpus_bleu_t( predictions[ pred_lang ],
                                            references[ pred_lang ],
                                            ref_weights= reference_weights,
                                            tokenize='zh',
                                            lowercase=True,
                                            use_effective_order=True )
            bert_scores=[]
            # for every post, the BERTScore is the maximum pairwise score of the prediction to the response
            for p,r in zip( predictions[ pred_lang ], references[ pred_lang ]):
                p=[p]*len(r)
                (P, R, F), _=score(r, p, lang="ch", return_hash=True)
                bert_scores.append(max([float(fi) for fi in F]))
        else :
            delatbleu = sacrebleu_deltableu.corpus_bleu_t( predictions[ pred_lang ],
                                            references[ pred_lang ],
                                            ref_weights= reference_weights,
                                            lowercase=True,
                                            use_effective_order=True )
            
            bert_scores=[]

            # for every post, the BERTScore is the maximum pairwise score of the prediction to the response
            for p,r in zip( predictions[ pred_lang ], references[ pred_lang ]):
                p=[p]*len(r)
                (P, R, F), _=score(r, p,lang="en",return_hash=True)
                bert_scores.append(max([float(fi) for fi in F]))

        #print( delatbleu )
        scores[ 'deltableu_{}'.format( pred_lang) ] = delatbleu.score
        scores[ 'BERTScore_{}'.format( pred_lang) ] = np.mean(bert_scores)
        #scores[ 'BERTScore_{}_by_post'.format( pred_lang) ] = bert_scores

    #print(scores)

    with open( score_path, 'w') as score_file:
        score_file.write(json.dumps(scores,indent=4))


if __name__ == "__main__":

    reference_dir = os.path.join('/app/input/', 'ref')
    prediction_dir = os.path.join('/app/input/', 'res')
    score_dir = '/app/output/'

    print('Reading reference dataset')
    truth = read_json( os.path.join(reference_dir, 'reference.json') )

    print('Reading prediction')
    prediction = read_json( os.path.join(prediction_dir, 'prediction.json') )
    score_path = os.path.join(score_dir, 'scores.json')

    df_userinfo = pd.read_csv( '{}/df_userinfo.csv'.format( reference_dir ) )

    main( truth, prediction, score_path, df_userinfo )

