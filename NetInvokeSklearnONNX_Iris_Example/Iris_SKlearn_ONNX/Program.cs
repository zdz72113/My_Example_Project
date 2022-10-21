using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

namespace Iris_SKlearn_ONNX
{
    internal class Program
    {
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "onnx_model", "rf_iris.onnx");
        static void Main(string[] args)
        {
            var outputColumnNames = new[] { "output_label", "output_probability" };
            var inputColumnNames = new[] { "float_input" };
            var shapeDictionary = new Dictionary<string, int[]>
                                    {
                                        { "float_input", new [] { 1, 4 } },
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
                Data = new[] { 6.0f, 2.2f, 4f, 1f }
            };
            var prediction = engine.Predict(iris);
            Console.WriteLine("预测结果：{0}", prediction.Prediction[0]);
        }
    }

    public class Iris
    {
        [VectorType(1 * 4)]
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