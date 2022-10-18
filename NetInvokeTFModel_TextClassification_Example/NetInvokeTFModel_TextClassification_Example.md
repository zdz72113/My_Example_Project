[TOC]

# 简介

本示例目的是验证 .Net 服务或应用程序调用 TensorFlow 训练出来的 pb 模型。
尽管 ML.Net 教程里面给出了两个 ML.NET 和 TensorFlow 的例子: [在 ML.NET 中使用预先训练的 TensorFlow 模型分析电影评论的情绪](https://learn.microsoft.com/zh-cn/dotnet/machine-learning/tutorials/text-classification-tf) & [通过 ML.NET 图像分类 API 使用迁移学习自动进行肉眼检查](https://learn.microsoft.com/zh-cn/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning)。但教程里面并没有告诉读者怎么去训练模型及查看 TensorFlow 模型的输入输出。

# 使用TensorFlow训练模型

参考[电影评论文本分类](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/keras/text_classification.ipynb), 使用存储在磁盘上的纯文本文件。训练一个二元分类器对 IMDB 数据集执行情感分析。最后保存模型到磁盘用于 .Net 调用。

# 查看模型输入输出

查看模型的输入输出可以采用两种方式：
- TensorFlow saved_model_cli 带的 saved_model_cli 命令或 [saved_model_cli.py](https://github.com/tensorflow/tensorflow/blob/c5da7af048611aa29e9382371f0aed5018516cac/tensorflow/python/tools/saved_model_cli.py) 脚本，命令如下：
```
python D:\Study\jupyter\Python_DA\saved_model_cli.py  show --dir D:\Study\jupyter\Python_DA\saved_model\imdb_model --all
```
- 通过工具 [netron](https://github.com/lutzroeder/netron)

# 创建.Net应用通过ML.Net调用模型

1. 创建.Net应用，引入Microsoft.ML.SampleUtils，Microsoft.ML.TensorFlow，SciSharp.TensorFlow.Redist，注意SciSharp.TensorFlow.Redist版本号，例子中使用的是2.3.1，参考[TensorFlow: Unable to find entry point 'TF_StringDecode'](https://github.com/dotnet/machinelearning/issues/6040)
2. 复制 TensorFlow 模型文件到项目目录，并对所有文件设置属性：将“复制到输出目录”的值更改为“如果较新则复制” 。
3. 编写代码并测试
```
MLContext mlContext = new MLContext();

//加载模型
TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);

var pipeline = tensorFlowModel.ScoreTensorFlowModel(outputColumnNames: new[] { "StatefulPartitionedCall:0" }, inputColumnNames: new[] { "serving_default_text_vectorization_input:0" }, addBatchDimensionInput: false);

IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<MovieReview>());
ITransformer model = pipeline.Fit(dataView);

//预测
var engine = mlContext.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);

var review = new MovieReview()
{
    ReviewText = new[] { "this film is really bad" }
};
var sentimentPrediction = engine.Predict(review);
Console.WriteLine("输入：{0}，预测结果：{1}", review.ReviewText[0], sentimentPrediction.Prediction[0] > 0.5 ? "积极" : "消极");
```
