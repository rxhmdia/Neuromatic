using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using FluentAssertions;
using Neuromatic.Initializers;
using TensorFlow;

namespace Neuromatic.Tests.Initializers
{
    public class RandomNormalTests
    {
        [Fact]
        public void GeneratesOutput()
        {
            using(var session = new TFSession())
            {
                var initializer = new RandomNormal().Compile(session.Graph,new TFShape(5,5));
                var output = session.GetRunner().Run(initializer);

                output.Shape[0].Should().Be(5);
                output.Shape[1].Should().Be(5);
            }
        }
    }
}
