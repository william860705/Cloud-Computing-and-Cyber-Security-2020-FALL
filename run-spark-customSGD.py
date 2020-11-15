
# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/spark" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path


from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import max, first, col, min
import pyspark

feats_min = []
feats_max = []

class CustomSGD:
    def __init__(self, train_data, lr, n_epoch):
        self.lr = lr
        self.w = np.zeros(train_data.first().size - 1)
        self.b = 0

        for epoch in range(n_epoch):
            train_data_random_subset = train_data.sample(False, 0.5, None)

            w = train_data_random_subset.map(lambda x: self.step(x)[0]).reduce(lambda x, y: x + y)
            b = train_data_random_subset.map(lambda x: self.step(x)[1]).reduce(lambda x, y: x + y)

            self.w = self.w + w / float(train_data_random_subset.count())
            self.b = self.b + b / float(train_data_random_subset.count())
            if epoch % 10 == 0: # print every 10 epochs
                print("Epoch " + str(epoch) + ":")
                print(self.w)
                print(self.b)

    def predict(self, x):
        yhat = np.asscalar(np.dot(self.w, x) + self.b)
        return 1.0 / (1.0 + np.exp(-yhat)) #sigmoid function

    def step(self, point):
        y = int(point.item(0))
        x = point.take(range(1, point.size))
        yhat = self.predict(x)
        error = y - yhat
        w = self.lr * (error * x)
        b = self.lr * error
        return w, b




def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    label = feats[-1]
    feats = feats[:-1]
    feats.insert(0,label) # put label at index 0
    features = [ float(feature) for feature in feats ] # need floats
    return np.array(features)

# sc = getSparkContext()
sc = pyspark.SparkContext()

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


# Load and parse the data
data = sc.textFile("data_banknote_authentication.txt")
parsedData = data.map(mapper)

model = CustomSGD(parsedData, lr=1, n_epoch=200)
labelsAndPreds = parsedData.map(lambda point: (int(point.item(0)), 
        round(model.predict(point.take(range(1, point.size))))))

acc = labelsAndPreds.filter(lambda (v, p): v == p).count() / float(parsedData.count())

print("Acc = " + str(acc))