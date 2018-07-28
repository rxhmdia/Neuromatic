using FluentAssertions;
using Neuromatic.Activations;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Activations
{
    public class RELUSpec
    {
        [Fact]
        public void ReturnsGraphElement()
        {
            var context = new ModelCompilationContext(new TFGraph());
            var input = context.Graph.Placeholder(TFDataType.Double, new TFShape(10, 10));
            var output = new RELU().Compile(context, input);

            output.Should().NotBeNull();
        }
    }
}