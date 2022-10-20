[TOC]

# 简介

本示例目的是验证 .Net 服务或应用程序调用 ONNX模型 （基于TensorFlow 训练出来的 pb 模型）。

# 使用TensorFlow训练模型

参考[Basic regression: Predict fuel efficiency](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/keras/regression.ipynb), 此教程使用经典的 Auto MPG 数据集并演示了如何构建模型来预测 20 世纪 70 年代末和 20 世纪 80 年代初汽车的燃油效率。保存模型：
```
model.save('saved_model\dnn_model')
```

# 转换模型为ONNX格式

```
# pip install git+https://github.com/onnx/tensorflow-onnx

python -m tf2onnx.convert --saved-model D:\Study\jupyter\Python_DA\saved_model\dnn_model --output dnn_model.onnx
```

# 查看模型输入输出

- 通过工具 [netron] 查看模型的输入输出 (https://github.com/lutzroeder/netron)

# 创建.Net应用通过ML.Net调用模型

1. 创建.Net应用，引入Microsoft.ML.OnnxRuntime，Microsoft.ML.OnnxTransformer
2. 复制 转换模型为ONNX格式 模型文件到项目目录，并对所有文件设置属性：将“复制到输出目录”的值更改为“如果较新则复制” 。
3. 编写代码并测试
```
internal class Program
{
    static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "onnx_model", "dnn_model.onnx");
    static void Main(string[] args)
    {
        var outputColumnNames = new[] { "dense_26" };
        var inputColumnNames = new[] { "normalization_6_input" };
        var shapeDictionary = new Dictionary<string, int[]>
                                {
                                    { "normalization_6_input", new [] { 1, 9 } },
                                    { "dense_26", new [] { 1, 1 } }
                                };
        MLContext mlContext = new MLContext();
        var pipeline = mlContext.Transforms.ApplyOnnxModel(outputColumnNames, inputColumnNames, _modelPath, shapeDictionary, null, false);
        IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<AutoMpg>());
        ITransformer model = pipeline.Fit(dataView);

        //预测
        var engine = mlContext.Model.CreatePredictionEngine<AutoMpg, AutoMpgPrediction>(model);

        var autoMpg = new AutoMpg()
        {
            Data = new[] { 4f, 120.0f, 79.0f, 2625.0f, 18.6f, 82f, 0f, 0f, 1f }
        };
        var prediction = engine.Predict(autoMpg);
        Console.WriteLine("预测结果：{0}", prediction.Prediction[0]);
    }
}

public class AutoMpg
{
    [VectorType(9)]
    [ColumnName("normalization_6_input")]
    public float[] Data { get; set; }
}

public class AutoMpgPrediction
{
    [VectorType(1)]
    [ColumnName("dense_26")]
    public float[] Prediction { get; set; }
}
```
