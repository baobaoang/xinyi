# Databricks notebook source
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
11122211

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC ## 哈哈哈哈啊a
# MAGIC

# COMMAND ----------

# 构造一个简单的DataFrame作为示例数据
data = [(1, 1), (2, 3), (3, 5), (4, 7), (5, 9)]  # 特征和标签数据
columns = ["feature", "label"]
df = spark.createDataFrame(data, columns)

# 使用VectorAssembler将特征转换为向量，因为许多ML算法要求输入数据为向量形式
assembler = VectorAssembler(inputCols=["feature"], outputCol="features_vector")
df_transformed = assembler.transform(df)

# 选择LinearRegression模型进行训练
lr = LinearRegression(featuresCol="features_vector", labelCol="label")

# 划分训练集（这里为了简化直接使用全部数据训练，实际应用中应划分为训练集和测试集）
train_data = df_transformed

# 训练模型
model = lr.fit(train_data)

# 打印模型参数
print("模型系数:", model.coefficients)
print("模型截距:", model.intercept)


# COMMAND ----------

# 使用模型进行预测（以第一行数据为例）
input_data = [[1]]  # 特征数据
input_df = spark.createDataFrame(input_data, ["feature"])
input_df = assembler.transform(input_df)
prediction = model.transform(input_df)
print("预测结果:", prediction.select("prediction").collect()[0][0])
