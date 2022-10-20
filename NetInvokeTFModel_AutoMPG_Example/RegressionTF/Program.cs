using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

namespace RegressionTF
{
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

    //internal class Program
    //{
    //    static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "tf_model", "frozen_dnn_model.pb");
    //    static void Main(string[] args)
    //    {
    //        MLContext mlContext = new MLContext();

    //        //加载模型
    //        TensorFlowModel tensorFlowModel = mlContext.Model.LoadTensorFlowModel(_modelPath);
    //        var aa = tensorFlowModel.GetInputSchema();

    //        var inputSchemaDefinition = SchemaDefinition.Create(typeof(AutoMpg));
    //        inputSchemaDefinition["Data"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 9);
    //        inputSchemaDefinition["Data"].ColumnName = "x";

    //        var outputSchemaDefinition = SchemaDefinition.Create(typeof(AutoMpgPrediction));
    //        outputSchemaDefinition["Prediction"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, 1);
    //        outputSchemaDefinition["Prediction"].ColumnName = "Identity";

    //        var pipeline = tensorFlowModel.ScoreTensorFlowModel(outputColumnName: "Identity", inputColumnName: "x", addBatchDimensionInput: false);

    //        IDataView dataView = mlContext.Data.LoadFromEnumerable(new List<AutoMpg>(), inputSchemaDefinition);
    //        ITransformer model = pipeline.Fit(dataView);

    //        //预测
    //        var engine = mlContext.Model.CreatePredictionEngine<AutoMpg, AutoMpgPrediction>(model, inputSchemaDefinition: inputSchemaDefinition, outputSchemaDefinition: outputSchemaDefinition);

    //        var review = new AutoMpg()
    //        {
    //            Data = new[] { 4f, 120.0f, 79.0f, 2625.0f, 18.6f, 82f, 0f, 0f, 1f }
    //        };
    //        var prediction = engine.Predict(review);
    //        //Console.WriteLine("输入：{0}，预测结果：{1}", review.ReviewText[0], sentimentPrediction.Prediction[0] > 0.5 ? "积极" : "消极");
    //    }
    //}

    //public class AutoMpg
    //{
    //    [VectorType(1, 9)]
    //    public float[] Data { get; set; }
    //}

    //public class AutoMpgPrediction
    //{
    //    [VectorType(1, 1)]
    //    public float[] Prediction { get; set; }
    //}
}