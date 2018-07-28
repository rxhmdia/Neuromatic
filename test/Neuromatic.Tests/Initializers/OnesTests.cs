using FluentAssertions;
using Neuromatic.Initializers;
using TensorFlow;
using Xunit;

namespace Neuromatic.Tests.Initializers
{
    public class OnesTests
    {
        [Fact]
        public void GeneratesOutput()
        {
            using(var session = new TFSession())
            {
                var initializer = new Ones().Compile(session.Graph,new TFShape(5,5));
                var output = session.GetRunner().Run(initializer);

                output.Shape[0].Should().Be(5);
                output.Shape[1].Should().Be(5);
            }
        }
    }
}