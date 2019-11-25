from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas



def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        # train_module
        vectorize_text = vectorizer.fit_transform(train_data.message_body)
        classifier.fit(vectorize_text, train_data.status)

        # score
        vectorize_text = vectorizer.transform(test_data.message_body)
        score = classifier.score(vectorize_text, test_data.status)
        string_score = ''
        string_score +=  str(score * 100)
        print(string_score)

def run_classifier(filename):
    data = pandas.read_csv(filename)
    learn = data[:600]
    test = data[600:]

    perform(
        [
            BernoulliNB(),
            RandomForestClassifier(n_estimators=100, n_jobs=-1),
            AdaBoostClassifier(),
            BaggingClassifier(),
            ExtraTreesClassifier(),
            GradientBoostingClassifier(),
            DecisionTreeClassifier(),
            CalibratedClassifierCV(),
            DummyClassifier(),
            PassiveAggressiveClassifier(),
            RidgeClassifier(),
            RidgeClassifierCV(),
            SGDClassifier(),
            OneVsRestClassifier(SVC(kernel='linear')),
            OneVsRestClassifier(LogisticRegression()),
            KNeighborsClassifier()
        ],
        [
            CountVectorizer(),
            TfidfVectorizer(),
            HashingVectorizer()
        ],
        learn,
        test
    )

