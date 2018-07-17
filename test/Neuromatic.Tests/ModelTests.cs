using FluentAssertions;
using Neuromatic.Layers;
using Neuromatic.Losses;
using Neuromatic.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Neuromatic.Tests
{
    public class ModelTests
    {
        [Fact]
        public void CanTrainForMultipleEpochs()
        {
            var input = new Input(new long[] { 2 }, "Input0");
            var dense = new Dense(2, input, name: "Dense0");

            var model = new Model(new[] { input }, new[] { dense });

            model.Compile(new SGD(0.001), new[] { new MeanSquaredError() });

            var features = new double[][]
            {
                new double[] { 5.0, 2.0 },
                new double[] { 1.0, 1.0 },
            };

            var labels = new double[][]
            {
                new double[] { 0.0, 1.0 },
                new double[] { 1.0, 0.0 }
            };

            for (int epoch = 0; epoch < 5; epoch++)
            {
                model.TrainMinibatch(
                    new Dictionary<Input, Array>
                    {
                        { input, features }
                    },
                    new Dictionary<Layer, Array>
                    {
                        { dense, labels }
                    });
            }
        }
        
        [Fact]
        public void CanPredict()
        {
            var input = new Input(new long[] { 2 }, "Input0");
            var dense = new Dense(2, input, name: "Dense0");

            var model = new Model(new[] { input }, new[] { dense });

            model.Compile(new SGD(0.001), new[] { new MeanSquaredError() });

            var features = new double[][]
            {
                new double[] { 5.0, 2.0 },
                new double[] { 1.0, 1.0 },
            };

            var labels = new double[][]
            {
                new double[] { 0.0, 1.0 },
                new double[] { 1.0, 0.0 }
            };

            var output = model.Predict(new Dictionary<Input, Array>
            {
                { input, features }
            });

            output[dense].Length.Should().Be(4);
        }
    }
}
