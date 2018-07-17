using System.Linq;
using FluentAssertions;
using Neuromatic.Layers;
using Neuromatic.Losses;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Losses
{
    public class CategoricalCrossEntropyTests
    {
        [Fact]
        public void ShouldCompile()
        {
            var graph = new TFGraph();
            var context = new ModelCompilationContext(graph);

            var predictions = graph.Placeholder(TFDataType.Double, new TFShape(-1, 10));
            var actuals = graph.Placeholder(TFDataType.Double, new TFShape(-1, 10));

            var loss = new CategoricalCrossEntropy().Compile(context, predictions, actuals);

            loss.Should().NotBeNull();
        }

        [Fact]
        public void ShouldBeOptimizable()
        {
            var graph = new TFGraph();
            var context = new ModelCompilationContext(graph);

            var input = new Input(new long[] { 10 }, name: "Input0");
            var output = new Dense(2, input, name: "Dense0");

            var compiledInput = input.Compile(context);
            var compiledOutput = output.Compile(context);

            var loss = new CategoricalCrossEntropy();

            var compiledLoss = loss.Compile(context, compiledOutput,
                context.Graph.Placeholder(TFDataType.Double, new TFShape(-1, 2)));

            var gradients = graph.AddGradients(
                new [] { compiledLoss },
                context.Parameters.ToArray());
        }
    }
}