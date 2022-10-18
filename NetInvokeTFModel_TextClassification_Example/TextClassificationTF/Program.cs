using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.IO;

namespace TextClassificationTF2
{
    internal class Program
    {
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");
        static void Main(string[] args)
        {
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
        }
    }

    public class MovieReview
    {
        [VectorType(1)]
        [ColumnName("serving_default_text_vectorization_input:0")]
        public string[] ReviewText { get; set; }
    }

    public class MovieReviewSentimentPrediction
    {
        [VectorType(1)]
        [ColumnName("StatefulPartitionedCall:0")]
        public float[] Prediction { get; set; }
    }
}