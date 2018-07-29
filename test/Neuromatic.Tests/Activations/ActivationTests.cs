using System;
using FluentAssertions;
using Neuromatic.Activations;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Activations
{
    public class ActivationTests
    {
        [Theory]
        [InlineData(typeof(Linear))]
        [InlineData(typeof(RELU))]
        [InlineData(typeof(Sigmoid))]
        [InlineData(typeof(Softmax))]
        [InlineData(typeof(Tanh))]
        public void ReturnsGraphElement(Type functionType)
        {
            var activation = (ActivationFunction)Activator.CreateInstance(functionType);

            var context = new ModelCompilationContext(new TFGraph());
            var input = context.Graph.Placeholder(TFDataType.Double, new TFShape(10, 10));
            var output = activation.Compile(context, input);

            output.Should().NotBeNull();
        }

        [Theory]
        [InlineData(typeof(Linear))]
        [InlineData(typeof(RELU))]
        [InlineData(typeof(Sigmoid))]
        [InlineData(typeof(Softmax))]
        [InlineData(typeof(Tanh))]
        public void ProducesOutput(Type functionType)
        {
            var activation = (ActivationFunction)Activator.CreateInstance(functionType);

            var graph = new TFGraph();
            var context = new ModelCompilationContext(graph);
            var input = context.Graph.Placeholder(TFDataType.Double, new TFShape(1));
            var output = activation.Compile(context, input);

            using (var session = new TFSession(graph))
            {
                var runner = session.GetRunner();

                runner.AddInput(input, new TFTensor(new[] { new[] { 1.0 } }));
                var outputValue = runner.Run(output);

                outputValue.GetValue().Should().NotBe(new[] { new[] { 0.0 } });
            }
        }
    }
}