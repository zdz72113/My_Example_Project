[TOC]

# 简介

本示例目的是验证 .Net 服务或应用程序调用 TensorFlow 训练出来的 pb 模型。验证多个不同数据类型输入的处理方法。

# 使用TensorFlow训练模型

参考[使用 Keras 预处理层对结构化数据进行分类](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/structured_data/preprocessing_layers.ipynb), 使用 PetFinder.my mini 数据集，预测宠物是否会被领养。最后保存模型到磁盘用于 .Net 调用。
```
model.save('my_pet_classifier')
```

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
MLContext mlContext = new MLContext(seed: 1);

TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);

var pipeline = tensorFlowModel.ScoreTensorFlowModel(outputColumnNames: new[] { "StatefulPartitionedCall" }, inputColumnNames: new[] { "serving_default_Age", "serving_default_Breed1",
        "serving_default_Color1", "serving_default_Color2", "serving_default_Fee", "serving_default_FurLength", "serving_default_Gender", "serving_default_Health",
        "serving_default_MaturitySize", "serving_default_PhotoAmt", "serving_default_Sterilized", "serving_default_Type", "serving_default_Vaccinated" }, addBatchDimensionInput: false);

IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<Pet>());
ITransformer model = pipeline.Fit(dataView);

var engine = mlContext.Model.CreatePredictionEngine<Pet, PetAdoptedPrediction>(model);

var review = new Pet()
{
    Age = new Int64[] { 3L  },
    Breed1 = new string[] { "Tabby" },

    Color1 = new string[] {  "Black" },
    Color2 = new string[] { "White"  },
    Fee = new float[] { 100f },
    FurLength = new string[] { "Short" },
    Gender = new string[] { "Male" },
    Health = new string[] { "Healthy" },
    MaturitySize = new string[] { "Small" },
    PhotoAmt = new float[] { 2f },
    Sterilized = new string[] { "No" },
    Type = new string[] { "Cat" },
    Vaccinated = new string[] { "No" },
};
var prediction = engine.Predict(review);
```
