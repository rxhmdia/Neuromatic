using FluentAssertions;
using Neuromatic.Losses;
using TensorFlow;

namespace Neuromatic.Tests.Losses 
{
    public class CategoricalCrossEntropyTests
    {
        public void ShouldCompile()
        {
            var graph = new TFGraph();

            var predictions = graph.Placeholder(TFDataType.Double, new TFShape(-1,10));
            var actuals = graph.Placeholder(TFDataType.Double, new TFShape(-1,10));

            var loss = new CategoricalCrossEntropy().Compile(graph, predictions,actuals);

            loss.Should().NotBeNull();
        }
    }
}