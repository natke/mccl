using System;
using System.IO;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;


    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        public static IEstimator<ITransformer> ProcessData(MLContext mlContext)
        {
            return mlContext.Transforms.Conversion.MapValueToKey("Area", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("Title", "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText("Description", "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);
        }

        public static ITransformer BuildAndTrainModel(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView trainingData)
        {
            var trainer = new SdcaMultiClassTrainer(mlContext, DefaultColumnNames.Label, DefaultColumnNames.Features);

            var trainingPipeline = pipeline.Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedModel = trainingPipeline.Fit(trainingData);

            SaveModelAsFile(mlContext, trainedModel);

            return trainedModel;
        }


        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }


        public static void Evaluate(MLContext mlContext, ITransformer trainedModel)
        {
            var testDataView = mlContext.Data.CreateTextReader<GitHubIssue>(hasHeader: true).Read(_testDataPath);

            var testMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private static void PredictIssue(MLContext mlContext)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };

            var predEngine = loadedModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(mlContext);

            var prediction = predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext(seed: 0);

            IDataView trainingData = mlContext.Data.CreateTextReader<GitHubIssue>(hasHeader: true).Read(_trainDataPath);

            var preProcessingPipeline = ProcessData(mlContext);

            var trainedModel = BuildAndTrainModel(mlContext, preProcessingPipeline, trainingData);

            Evaluate(mlContext, trainedModel);

            PredictIssue(mlContext);

        }
    }
