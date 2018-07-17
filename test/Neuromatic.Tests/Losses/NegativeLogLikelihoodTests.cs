using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FluentAssertions;
using Neuromatic.Layers;
using Neuromatic.Losses;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Losses
{
    public class NegativeLogLikelihoodTests
    {
        [Fact]
        public void CanBeOptimized()
        {
            var graph = new TFGraph();
            var context = new ModelCompilationContext(graph);

            var input = new Input(new long[] { 10 }, name: "Input0");
            var output = new Dense(2, input, name: "Dense0");

            var compiledOutput = output.Compile(context);

            var loss = new NegativeLogLikelihood();

            var compiledLoss = loss.Compile(context, compiledOutput,
                context.Graph.Placeholder(TFDataType.Double, new TFShape(-1, 2)));

            var gradients = graph.AddGradients(
                new[] { compiledLoss },
                context.Parameters.ToArray());
        }

        [Fact]
        public void SolvesTowardsZeroForCorrectCases()
        {
            var graph = new TFGraph();
            var context = new ModelCompilationContext(graph);

            var output = graph.Const(new[] { new[] { 1.0, 0.0 } });
            var targets = graph.Const(new[] { new[] { 1.0, 0.0 } });

            var loss = new NegativeLogLikelihood();

            var lossFunction = loss.Compile(context, output, targets);

            using (var session = new TFSession(graph))
            {
                var error = session.GetRunner().Run(lossFunction);

                Math.Round((double)error.GetValue(), 2).Should().Be(0.0);
            }
        }

        [Fact]
        public void ProducesHighLossForIncorrectCases()
        {
            var graph = new TFGraph();
            var context = new ModelCompilationContext(graph);

            var output = graph.Const(new[] { new[] { 0.0, 0.1 } });
            var targets = graph.Const(new[] { new[] { 1.0, 0.0 } });

            var loss = new NegativeLogLikelihood();

            var lossFunction = loss.Compile(context, output, targets);

            using (var session = new TFSession(graph))
            {
                var error = session.GetRunner().Run(lossFunction);
                Math.Round((double)error.GetValue(), 2).Should().Be(16.12);
            }
        }
    }
}
