using FakeItEasy;
using FluentAssertions;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Layers
{
    public class InputTests
    {
        [Fact]
        public void CreatesLayerConfigurationDuringCompilation()
        {
            var input = new Input(new long[] { 10 });
            var graph = new TFGraph();

            input.Compile(new ModelCompilationContext(graph));

            input.Configuration.Should().NotBeNull();
            input.Configuration.Output.Should().NotBeNull();
            input.Configuration.Parameters.Count().Should().Be(0);
            input.Configuration.Initializers.Count().Should().Be(0);
        }

        [Fact]
        public void MustHaveOneOrMoreDimensions()
        {
            var input = new Input(new long[] { });
            var graph = new TFGraph();

            input.Invoking(x => x.Compile(new ModelCompilationContext(graph))).Should().Throw<ModelCompilationException>();
        }

        [Fact]
        public void IncludesExtraDimensionInOutputShape()
        {
            var input = new Input(new long[] {  10});

            input.OutputShape.Should().BeEquivalentTo(new long[] { -1, 10 });
        }
    }
}
