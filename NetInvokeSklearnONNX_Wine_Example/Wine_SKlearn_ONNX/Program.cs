using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

namespace Wine_SKlearn_ONNX
{
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
}