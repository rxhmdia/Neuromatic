using Neuromatic.Core;
using Neuromatic.Layers;
using Neuromatic.Metrics;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Neuromatic.TensorFlow.Tests
{
    public class FeedForwardNetworkTests
    {
        [Fact]
        public void CompilesSuccesfully()
        {
            using (ModelBackend backend = new TensorFlowModelBackend())
            {
                var inputLayer = new Input(new long[] { -1, 20 });
                var outputLayer = new Dense(2, StandardActivations.Sigmoid(), inputLayer, "dense-1");

                var model = new Model(
                    new [] {  inputLayer}, 
                    new [] { outputLayer}, 
                    new[] { StandardLosses.BinaryCrossEntropy() }, 
                    StandardOptimizers.SGD(), 
                    new MetricFunction[] {  });

                model.Compile(backend);
            }
        }
    }
}
