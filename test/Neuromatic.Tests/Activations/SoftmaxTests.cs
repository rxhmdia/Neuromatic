using FluentAssertions;
using Neuromatic.Activations;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Activations
{
    public class SoftmaxTests
    {
        [Fact]
        public void ReturnsGraphElement()
        {
            var context = new ModelCompilationContext(new TFGraph());
            var input = context.Graph.Placeholder(TFDataType.Double, new TFShape(10, 10));
            var output = new Softmax().Compile(context, input);

            output.Should().NotBeNull();
        }
    }
}