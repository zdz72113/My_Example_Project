using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace PetFinderTF
{
    internal class Program
    {
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "tf_model");
        static void Main(string[] args)
        {
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
        }
    }

    public class Pet
    {
        [VectorType(1)]
        [ColumnName("serving_default_Age")]
        public Int64[] Age { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Breed1")]
        public string[] Breed1 { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Color1")]
        public string[] Color1 { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Color2")]
        public string[] Color2 { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Fee")]
        public float[] Fee { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_FurLength")]
        public string[] FurLength { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Gender")]
        public string[] Gender { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Health")]
        public string[] Health { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_MaturitySize")]
        public string[] MaturitySize { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_PhotoAmt")]
        public float[] PhotoAmt { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Sterilized")]
        public string[] Sterilized { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Type")]
        public string[] Type { get; set; }

        [VectorType(1)]
        [ColumnName("serving_default_Vaccinated")]
        public string[] Vaccinated { get; set; }
    }

    public class PetAdoptedPrediction
    {
        [VectorType(1)]
        [ColumnName("StatefulPartitionedCall")]
        public float[] Prediction { get; set; }
    }
}