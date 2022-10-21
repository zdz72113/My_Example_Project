[TOC]

# 简介

本示例目的是验证 .Net 服务或应用程序调用 ONNX模型 （基于Sklearn训练）。

# 使用Sklearn训练模型并保存为ONNX格式

数据集使用[wine](https://scikit-learn.org/stable/datasets/toy_dataset.html)，特征缩放参考[特征缩放的重要性](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py)，ONNX保存参考http://onnx.ai/sklearn-onnx/introduction.html


# 查看模型输入输出

- 通过工具 netron 查看模型的输入输出 (https://netron.app/)

# 创建.Net应用通过ML.Net调用模型

1. 创建.Net应用，引入Microsoft.ML.OnnxRuntime，Microsoft.ML.OnnxTransformer
2. 复制 转换模型为ONNX格式 模型文件到项目目录，并对所有文件设置属性：将“复制到输出目录”的值更改为“如果较新则复制” 。
3. 编写代码并测试
```
internal class Program
{
    static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "onnx_model", "GNB_wine.onnx");
    static void Main(string[] args)
    {
        var outputColumnNames = new[] { "output_label", "output_probability" };
        var inputColumnNames = new[] { "float_input" };
        var shapeDictionary = new Dictionary<string, int[]>
                                {
                                    { "float_input", new [] { 1, 13 } },
                                    { "output_label", new [] { 1 } },
                                };
        MLContext mlContext = new MLContext();
        var pipeline = mlContext.Transforms.ApplyOnnxModel(outputColumnNames, inputColumnNames, _modelPath, shapeDictionary, null, false);
        IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<Iris>());
        ITransformer model = pipeline.Fit(dataView);

        //预测
        var engine = mlContext.Model.CreatePredictionEngine<Iris, IrisPrediction>(model);

        var iris = new Iris()
        {
            Data = new[] { 13.71f, 5.65f, 2.45f, 20.5f, 95.0f, 1.68f, 0.61f, 0.52f, 1.06f, 7.70f, 0.64f, 1.74f, 740.0f }
        };
        var prediction = engine.Predict(iris);
        Console.WriteLine("预测结果：{0}", prediction.Prediction[0]);
    }
}

public class Iris
{
    [VectorType(1 * 13)]
    [ColumnName("float_input")]
    public float[] Data { get; set; }
}

public class IrisPrediction
{
    [VectorType(1)]
    [ColumnName("output_label")]
    public long[] Prediction { get; set; }

    [OnnxSequenceType(typeof(IDictionary<long, float>))]
    [ColumnName("output_probability")]
    public IEnumerable<IDictionary<long, float>> Probability { get; set; }
}
```
