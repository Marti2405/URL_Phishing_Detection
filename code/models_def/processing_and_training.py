# from url_utils import custom_preprocessor, custom_tokenizer
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import math
from collections import Counter
from sklearn.preprocessing import QuantileTransformer
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





##############################################################
##############################################################
#### Data processing
##############################################################
##############################################################


print("Starting the data processing.")

df = pd.read_csv('./data/urls.csv') # 0->legitimate # 1->phishing

print("0/2")

nb_lgtm = df['label'].value_counts()[0] / df.__len__() *100
nb_phi = df['label'].value_counts()[1] / df.__len__() *100

# print(f"Percentage of legitimate urls : {nb_lgtm.round(2)}%\nPercentage of phishing urls : {nb_phi.round(2)}%")



df['length'] = df['text'].apply(lambda x: len(x))



def get_hostname(text):
    x = text
    x = re.sub('^https://', '', x)
    x = re.sub('^http://', '', x)
    x = x.split('/')
    return x[0]

df['hostname'] = df['text'].apply(get_hostname)
df['hostnameLength'] = df['hostname'].apply(lambda x: len(x))
df['hostnameLengthRatio'] = df.apply(lambda x: x['hostnameLength']/x['length'] if x['length'] != 0 else 0, axis=1)



def get_path(row):
    try:
        x = row['text'].split(row['hostname'])[-1]
    except: # the hostname is an empty string
        # print(f"Hostname not found for the url \"{row['text']}\"")
        x = row['text']
        x = re.sub('^https://', '', x)
        x = re.sub('^http://', '', x)
        # print(f'Path found -> {x}')
    return x

df['path'] = df.apply(get_path,axis=1)
df['pathLength'] = df['path'].apply(lambda x: len(x))
df['pathLengthRatio'] = df.apply(lambda x: x['pathLength']/x['length'] if x['length'] != 0 else 0 , axis=1)

print("1/2")

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

print("2/2")




def calc_entropy(url):
    char_counts = Counter(url) # get dict of all the caracter counts in the url

    total_chars = len(url)

    if total_chars==0:
        return 0
    
    probabilities = [count / total_chars for count in char_counts.values()] # get probabilities or each char in the url

    entropy = -sum(p*math.log2(p) for p in probabilities) # Shannon entropy

    return entropy



df['urlEntropy'] = df['text'].apply(calc_entropy)



##############################################################
##############################################################
#### Train multinomialNB
##############################################################
##############################################################


print('Training MultinomialNB...')

# Create a pipeline
vectorizer = CountVectorizer()
classifier = MultinomialNB()

vect_and_multinomialNB = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


# Train the pipeline on the training data
vect_and_multinomialNB.fit(X_train, y_train)

# Predict the labels on the test data
y_pred = vect_and_multinomialNB.predict(X_test)


print('MultinomialNB:')
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save the pipeline
joblib.dump(vect_and_multinomialNB, './models_saved/vectorizer_and_multinomialNB.joblib')

print("Saved MultinomialNB successfully.")



# Function to score a URL list
def Bayes_score_url(url_list):
    
    # Predict the score
    score = vect_and_multinomialNB.predict_proba(url_list)[:,1] # Probability of phishing
    return score


print("0/1")
df['bayesProba'] = Bayes_score_url(df['text'].to_list())
print("1/1")

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




##############################################################
##############################################################
#### Train quantile transformer
##############################################################
##############################################################

X_train, X_test, y_train, y_test = train_test_split(df[cols], df['label'], test_size=0.2, random_state=42)

quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=0)

quantile_transformer.fit(X_train)

# Save the fitted scaler to a file
joblib.dump(quantile_transformer, './models_saved/quantile_transformer.joblib')



# process the data
df[cols] = quantile_transformer.transform(df[cols])



df.to_csv('./data/processed_data.csv', index=False)

print("Data processing done.")



##############################################################
##############################################################
#### Train random forest
##############################################################
##############################################################


print('Training RandomForest.')


X_train, X_test, y_train, y_test = train_test_split(df[cols], df['label'], test_size=0.2, random_state=42)


print('1/2')


rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=60)
rf_model.fit(X_train, y_train)

print('2/2')

print("Random Forest:")

y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))



joblib.dump(rf_model,'./models_saved/random_forest.joblib')

print("Saved Random Forest successfully.")