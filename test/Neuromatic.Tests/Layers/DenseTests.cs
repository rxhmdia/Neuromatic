using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;
using Xunit;
using FluentAssertions;
using System.Linq;

namespace Neuromatic.Tests.Layers
{
    public class DenseTests
    {
        [Fact]
        public void CreatesLayerConfigurationDuringCompilation()
        {
            var graph = new TFGraph();
            var input = new Input(new[] { 10L });
            
            var dense = new Dense(2, input, name: "Dense0");

            dense.Compile(new ModelCompilationContext(graph));

            dense.Configuration.Initializers.Count().Should().Be(2);
            dense.Configuration.Parameters.Count().Should().Be(2);
            dense.Configuration.Output.Should().NotBeNull();
        }

        [Fact]
        public void OutputShapeIsEqualToKernelSize()
        {
            var graph = new TFGraph();
            var input = new Input(new[] { 10L });

            input.Compile(new ModelCompilationContext(graph));

            var dense = new Dense(2, input, name: "Dense0");

            dense.Compile(new ModelCompilationContext(graph));

            dense.OutputShape.Should().BeEquivalentTo(new long[] { -1, 2 });
        }
    }

}
