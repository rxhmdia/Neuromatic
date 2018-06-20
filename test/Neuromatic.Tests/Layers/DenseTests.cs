using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;
using Xunit;
using FluentAssertions;

namespace Neuromatic.Tests.Layers
{
    public class DenseTests
    {
        [Fact]
        public void CreatesLayerConfigurationDuringCompilation()
        {
            var graph = new TFGraph();
            var input = new Input(new[] { 10L });
            
            var dense = new Dense(2, input,"Dense0");

            dense.Compile(graph);

            dense.Configuration.Initializers.Length.Should().Be(2);
            dense.Configuration.Parameters.Length.Should().Be(2);
            dense.Configuration.Output.Should().NotBeNull();
        }

        [Fact]
        public void OutputShapeIsEqualToKernelSize()
        {
            var graph = new TFGraph();
            var input = new Input(new[] { 10L });

            input.Compile(graph);

            var dense = new Dense(2, input, "Dense0");

            dense.OutputShape.Should().BeEquivalentTo(new long[] { -1, 2 });
        }
    }

}
