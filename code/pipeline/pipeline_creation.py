##############################################
# 
# RUN from main directory
#  
# 
##############################################

from sklearn.base import BaseEstimator
import dill
import joblib



class urlPredictor(BaseEstimator):

    def __init__(self, vect_and_mNB, quantile_transformer, random_forest):
        self.vect_and_mNB = vect_and_mNB
        self.quantile_transformer = quantile_transformer
        self.random_forest = random_forest
    

    def _get_hostname(self,text):
        import re
        x = text
        x = re.sub('^https://', '', x)
        x = re.sub('^http://', '', x)
        x = x.split('/')
        return x[0]
    
    def _get_path(self, row):
        import re
        try:
            x = row['text'].split(row['hostname'])[-1]
        except: # the hostname is an empty string
            print(f"Hostname not found for the url \"{row['text']}\"")
            x = row['text']
            x = re.sub('^https://', '', x)
            x = re.sub('^http://', '', x)
            print(f'Path found -> {x}')
        return x
    

    def _calc_entropy(self, url):
        from collections import Counter
        import math
        char_counts = Counter(url) # get dict of all the caracter counts in the url

        total_chars = len(url)

        if total_chars==0:
            return 0
        
        probabilities = [count / total_chars for count in char_counts.values()] # get probabilities or each char in the url

        entropy = -sum(p*math.log2(p) for p in probabilities) # Shannon entropy

        return entropy
    

    def predict(self, X):
        
        import pandas as pd
        import re
        import numpy as np

        df = pd.DataFrame({'text':X}) # Convert to dataframe for vectorized operations
        cols = [
                'hyphenRatio',
                'nb_dot',
                'length',
                'pathLengthRatio',
                'urlEntropy',
                'slashRatio',
                'numbersToLettersRatio',
                'numbersRatio',
                'hostnameLength',
                'nb_specialcaracters',
                'nb_hyphen',
                'bayesProba',
                'dotRatio',
                'nb_letters',
                'nb_slash',
                'lettersRatio',
                'specialcaractersRatio',
                'hostnameLengthRatio',
                'nb_numbers',
                'pathLength'
        ]


        ###############################################

        df['bayesProba'] = self.vect_and_mNB.predict_proba(df["text"])[:,1] # vectorize the url and predict the probability of phishing with MultinomialNB

        ###############################################

        df['length'] = df['text'].apply(lambda x: len(x))

        ###############################################
    

        df['hostname'] = df['text'].apply(self._get_hostname)
        df['hostnameLength'] = df['hostname'].apply(lambda x: len(x))
        df['hostnameLengthRatio'] = df.apply(lambda x: x['hostnameLength']/x['length'] if x['length'] != 0 else 0, axis=1)

        ###############################################



        df['path'] = df.apply(self._get_path,axis=1)
        df['pathLength'] = df['path'].apply(lambda x: len(x))
        df['pathLengthRatio'] = df.apply(lambda x: x['pathLength']/x['length'] if x['length'] != 0 else 0 , axis=1)

        #############
        df['nb_slash'] = df['text'].apply(lambda x: x.count("/"))
        df['nb_dot'] = df['text'].apply(lambda x: x.count("."))
        df['nb_hyphen'] = df['text'].apply(lambda x: x.count("-"))
        df['nb_numbers'] = df['text'].apply(lambda x: len(re.sub("[^0-9]", "", x)))
        df['nb_letters'] = df['text'].apply(lambda x: len(re.sub("[^a-zA-Z]", "", x)))
        df['nb_specialcaracters'] = df['text'].apply(lambda x: len(re.sub("[\w]+", "", x)))

        df['slashRatio'] = df.apply(lambda x: x['nb_slash']/x['length'] if x['length']!=0 else 0 ,axis=1)
        df['dotRatio'] = df.apply(lambda x: x['nb_dot']/x['length'] if x['length']!=0 else 0 ,axis=1)
        df['hyphenRatio'] = df.apply(lambda x: x['nb_hyphen']/x['length'] if x['length']!=0 else 0 ,axis=1)
        df['numbersRatio'] = df.apply(lambda x: x['nb_numbers']/x['length'] if x['length']!=0 else 0 ,axis=1)
        df['lettersRatio'] = df.apply(lambda x: x['nb_letters']/x['length'] if x['length']!=0 else 0 ,axis=1)
        df['specialcaractersRatio'] = df.apply(lambda x: x['nb_specialcaracters']/x['length'] if x['length']!=0 else 0 ,axis=1)
        df['numbersToLettersRatio'] = df.apply(lambda x: x['nb_numbers']/x['nb_letters'] if x['nb_letters']!=0 else 0 ,axis=1)


        ###############################################

        df['urlEntropy'] = df['text'].apply(self._calc_entropy)

        ###############################################

        df[cols] = self.quantile_transformer.transform(df[cols])

        ###############################################
        y = self.random_forest.predict_proba(df[cols])[:,1]

                
        
        return np.array(y)
    



if __name__=='__main__':
    

    vect_and_mNB = joblib.load('./models_saved/vectorizer_and_multinomialNB.joblib')
    quantile_transformer = joblib.load('./models_saved/quantile_transformer.joblib')
    random_forest = joblib.load('./models_saved/random_forest.joblib')

    
    url_predictor = urlPredictor(vect_and_mNB, quantile_transformer, random_forest)


    # res = joblib.dump(url_predictor, './pipeline/prediction_pipeline.joblib')
    
    # Save the entire class instance
    with open('url_predictor.dill', 'wb') as f:
        dill.dump(url_predictor, f)

    print("Created pipeline successfully")