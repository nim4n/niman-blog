# -*- coding: utf8 -*-
import numpy as np
from hazm import Stemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer



persian_stemmer = Stemmer()


class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (persian_stemmer.stem(w) for w in analyzer(doc))



count_vect = StemmedTfidfVectorizer(stop_words='persian')

train_data = ['پایتون زبان برنامه نویسی خوبی است', 'لینوکس یک سیستم عامل است', 'گیاهان دارویی را باید استفاده کرد', 'لینوکس یک سیستم عامل متن باز است' , 'پایتون زبان مناسبی برای یادگیری ماشینی است']    

target = np.array([1,2,3,2,1])

train_counts = count_vect.fit_transform(train_data)


clf = MultinomialNB().fit(train_counts, target)


test_data = ['با پایتون میتوان در لینوکس برنامه نویسی کرد', 'من لینوکس را دوست دارم']


test_counts = count_vect.transform(test_data)

predicted = clf.predict(test_counts)

for doc, category in zip(test_data, predicted):
     print('%s => %s' % (doc, category))


