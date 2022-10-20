[TOC]

# 简介

本示例目的是验证 .Net 服务或应用程序调用 TensorFlow 训练出来的 pb 模型。

# 使用TensorFlow训练模型

参考[Basic regression: Predict fuel efficiency](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/keras/regression.ipynb), 此教程使用经典的 Auto MPG 数据集并演示了如何构建模型来预测 20 世纪 70 年代末和 20 世纪 80 年代初汽车的燃油效率。保存模型：
```
model.save('saved_model\dnn_model')
```
在构建模型时候对原有模型进行了一定的改动，在归一化层之前明确的加入一层输入层，否则训练出来的模型.Net调用会报错。
```
normalizer = layers.Normalization()
normalizer.adapt(train_features)

# 构造输入层
inputs = keras.Input(shape=train_features.tail(1).shape[1:])
# 搭建网络各层
x = normalizer(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(1)(x)  # 输出结果是1个
# 构造模型
dnn_model = keras.Model(inputs, outputs)
# 查看模型结构
dnn_model.summary()
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
internal class Program
    {
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "tf_model");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 1);

            //加载模型
            TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);

            var pipeline = tensorFlowModel.ScoreTensorFlowModel(outputColumnNames: new[] { "StatefulPartitionedCall" }, inputColumnNames: new[] { "serving_default_input_6" }, addBatchDimensionInput: false);

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
        [VectorType(1, 9)]
        [ColumnName("serving_default_input_6")]
        public float[] Data { get; set; }
    }

    public class AutoMpgPrediction
    {
        [VectorType(1, 1)]
        [ColumnName("StatefulPartitionedCall")]
        public float[] Prediction { get; set; }
    }
```
