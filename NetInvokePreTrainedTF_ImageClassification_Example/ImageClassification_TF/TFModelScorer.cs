using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageClassification_TF
{
    public class TFModelScorer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly string labelsLocation;
        private readonly MLContext mlContext;
        private static string ImageReal = nameof(ImageReal);

        public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.labelsLocation = labelsLocation;
            mlContext = new MLContext();
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 224;
            public const float mean = 117;
            public const bool channelsLast = true;
        }

        public struct InceptionSettings
        {
            // for checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string inputTensorName = "input";

            // output tensor name
            public const string outputTensorName = "softmax2";
        }

        public void Score()
        {
            var model = LoadModel(dataLocation, imagesFolder, modelLocation);

            var predictions = PredictDataUsingModel(dataLocation, imagesFolder, labelsLocation, model).ToArray();

        }

        private PredictionEngine<ImageNetData, ImageNetPrediction> LoadModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}");

            var data = mlContext.Data.LoadFromTextFile<ImageNetData>(dataLocation, hasHeader: true);

            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(ImageNetData.ImagePath))
                            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "input"))
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageNetSettings.channelsLast, offsetImage: ImageNetSettings.mean))
                            .Append(mlContext.Model.LoadTensorFlowModel(modelLocation).
                            ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2" },
                                                inputColumnNames: new[] { "input" }, addBatchDimensionInput: true));

            ITransformer model = pipeline.Fit(data);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            return predictionEngine;
        }

        protected IEnumerable<ImageNetData> PredictDataUsingModel(string testLocation,
                                                                  string imagesFolder,
                                                                  string labelsLocation,
                                                                  PredictionEngine<ImageNetData, ImageNetPrediction> model)
        {
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            var labels = ModelHelpers.ReadLabels(labelsLocation);

            var testData = ImageNetData.ReadFromCsv(testLocation, imagesFolder);

            foreach (var sample in testData)
            {
                var probs = model.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedLabel, imageData.Probability) = ModelHelpers.GetBestLabel(labels, probs);
                imageData.ConsoleWrite();
                yield return imageData;
            }
        }

    }

    public static class ModelHelpers
    {
        static FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

        public static string GetAssetsPath(params string[] paths)
        {
            if (paths == null || paths.Length == 0)
                return null;

            return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());
        }

        public static string DeleteAssets(params string[] paths)
        {
            var location = GetAssetsPath(paths);

            if (!string.IsNullOrWhiteSpace(location) && File.Exists(location))
                File.Delete(location);
            return location;
        }

        public static (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);
            return (labels[index], max);
        }

        public static string[] ReadLabels(string labelsLocation)
        {
            return File.ReadAllLines(labelsLocation);
        }

        public static IEnumerable<string> Columns<T>() where T : class
        {
            return typeof(T).GetProperties().Select(p => p.Name);
        }

        public static IEnumerable<string> Columns<T, U>() where T : class
        {
            var typeofU = typeof(U);
            return typeof(T).GetProperties().Where(c => c.PropertyType == typeofU).Select(p => p.Name);
        }

        public static IEnumerable<string> Columns<T, U, V>() where T : class
        {
            var typeofUV = new[] { typeof(U), typeof(V) };
            return typeof(T).GetProperties().Where(c => typeofUV.Contains(c.PropertyType)).Select(p => p.Name);
        }

        public static IEnumerable<string> Columns<T, U, V, W>() where T : class
        {
            var typeofUVW = new[] { typeof(U), typeof(V), typeof(W) };
            return typeof(T).GetProperties().Where(c => typeofUVW.Contains(c.PropertyType)).Select(p => p.Name);
        }

        public static string[] ColumnsNumerical<T>() where T : class
        {
            return Columns<T, float, int>().ToArray();
        }

        public static string[] ColumnsString<T>() where T : class
        {
            return Columns<T, string>().ToArray();
        }

        public static string[] ColumnsDateTime<T>() where T : class
        {
            return Columns<T, DateTime>().ToArray();
        }
    }

    public static class ConsoleHelpers
    {
        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(" ");
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            Console.WriteLine(new String('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(" ");
            Console.WriteLine("Press any key to finish.");
            Console.ForegroundColor = defaultColor;
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            const string exceptionTitle = "EXCEPTION";

            Console.WriteLine(" ");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(exceptionTitle);
            Console.WriteLine(new String('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
        }

        public static void ConsoleWrite(this ImageNetDataProbability self)
        {
            var defaultForeground = Console.ForegroundColor;
            var labelColor = ConsoleColor.Magenta;
            var probColor = ConsoleColor.Blue;
            var exactLabel = ConsoleColor.Green;
            var failLabel = ConsoleColor.Red;

            Console.Write("ImagePath: ");
            Console.ForegroundColor = labelColor;
            Console.Write($"{Path.GetFileName(self.ImagePath)}");
            Console.ForegroundColor = defaultForeground;
            Console.Write(" labeled as ");
            Console.ForegroundColor = labelColor;
            Console.Write(self.Label);
            Console.ForegroundColor = defaultForeground;
            Console.Write(" predicted as ");
            if (self.Label.Equals(self.PredictedLabel))
            {
                Console.ForegroundColor = exactLabel;
                Console.Write($"{self.PredictedLabel}");
            }
            else
            {
                Console.ForegroundColor = failLabel;
                Console.Write($"{self.PredictedLabel}");
            }
            Console.ForegroundColor = defaultForeground;
            Console.Write(" with probability ");
            Console.ForegroundColor = probColor;
            Console.Write(self.Probability);
            Console.ForegroundColor = defaultForeground;
            Console.WriteLine("");
        }

    }
}
