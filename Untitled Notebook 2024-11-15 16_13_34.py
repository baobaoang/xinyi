from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

# 初始化SparkSession
spark = SparkSession.builder.appName("PySparkScikitLearnExample").getOrCreate()

# 读取CSV文件
df = spark.read.csv("path_to_your_csv_file.csv", header=True, inferSchema=True)

# 选择特征列并组合成一个向量列
assembler = VectorAssembler(inputCols=["feature1", "feature2", "feature3"], outputCol="features")
output = assembler.transform(df)
feature_df = output.select("features", "label")

# 分割数据集为训练集和测试集
train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=1234)

# 使用PySpark的LogisticRegression训练模型
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)

# 使用训练好的模型进行预测
predictions = lr_model.transform(test_df)

# 将PySpark DataFrame转换为本地数据集以使用scikit-learn
predictions_pd = predictions.toPandas()

# 使用scikit-learn的LogisticRegression进行预测
sk_model = SKLogisticRegression()
sk_model.fit(predictions_pd["features"], predictions_pd["prediction"])

# 打印scikit-learn模型的系数
print(sk_model.coef_)

# 停止SparkSession
spark.stop()
