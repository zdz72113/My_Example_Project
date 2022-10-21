using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageClassification_TF
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;

        public static IEnumerable<ImageNetData> ReadFromCsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(x => x.Split('\t'))
             .Select(x => new ImageNetData { ImagePath = Path.Combine(folder, x[0]), Label = x[1] });
        }
    }

    public class ImageNetDataProbability : ImageNetData
    {
        public string PredictedLabel;
        public float Probability { get; set; }
    }

    public class ImageNetPrediction
    {
        [ColumnName("softmax2")]
        public float[] PredictedLabels;
    }
}
