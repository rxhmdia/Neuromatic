using FluentAssertions;
using Neuromatic.Losses;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Losses
{
    public class MeanSquaredErrorTests
    {
        [Fact]
        public void ReturnsLossFunction()
        {
            var graph = new TFGraph();

            var predictions = graph.Placeholder(TFDataType.Double, new TFShape(-1,10));
            var actuals = graph.Placeholder(TFDataType.Double, new TFShape(-1,10));

            var loss = new MeanSquaredError().Compile(graph, predictions,actuals);

            loss.Should().NotBeNull();
        }
    }
}
