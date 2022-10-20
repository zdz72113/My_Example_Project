using Microsoft.ML;
using Microsoft.ML.Data;

namespace RegressionONNX
{
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
}