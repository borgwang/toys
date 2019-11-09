from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from naive_bayes import GaussianNaiveBayes
from naive_bayes import MultinomialNaiveBayes


def main():
    data = load_iris()
    x, y = data.data, data.target

    print("Gaussian Naive Bayes")
    model = GaussianNaiveBayes()
    model.fit(x, y)
    y_pred = model.predict(x)
    print("acc-mine: %.4f" % accuracy_score(y, y_pred))

    model = GaussianNB()
    model.fit(x, y)
    y_pred = model.predict(x)
    print("acc-sklearn: %.4f" % accuracy_score(y, y_pred))


    print("Multinomial Naive Bayes")
    model = MultinomialNaiveBayes()
    model.fit(x, y)
    y_pred = model.predict(x)
    print("acc-mine: %.4f" % accuracy_score(y, y_pred))

    model = MultinomialNB()
    model.fit(x, y)
    y_pred = model.predict(x)
    print("acc-sklearn: %.4f" % accuracy_score(y, y_pred))


if __name__ == "__main__":
    main()
