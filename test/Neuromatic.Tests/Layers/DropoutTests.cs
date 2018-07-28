using System.Linq;
using System.Runtime.Serialization;
using FluentAssertions;
using Neuromatic.Layers;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Layers
{
    public class DropoutTests
    {
        [Fact]
        public void CreatesLayerConfigurationDuringCompilation()
        {
            var context = new ModelCompilationContext(new TFGraph());

            var input = new Input(new [] { 10L });
            var layer = new Dropout(0.2, input);

            layer.Compile(context);

            layer.Configuration.Should().NotBeNull();
            layer.Configuration.Parameters.Count().Should().Be(0);
            layer.Configuration.Initializers.Count().Should().Be(0);
            layer.Configuration.Output.Should().NotBeNull();
        }

        [Fact]
        public void OutputShapeIsEqualToInputLayerShape()
        {
            var context = new ModelCompilationContext(new TFGraph());

            var input = new Input(new [] { 10L });
            var layer = new Dropout(0.2, input);

            layer.Compile(context);

            layer.OutputShape.Should().BeEquivalentTo(input.OutputShape);
        }
    }
}