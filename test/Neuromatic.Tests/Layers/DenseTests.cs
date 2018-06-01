using FakeItEasy;
using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;
using FluentAssertions;

namespace Neuromatic.Tests.Layers
{
    public class DenseTests
    {
        private ModelBackend CreateBackend()
        {
            var backend = A.Fake<ModelBackend>();
            var activations = A.Fake<Activations>();
            var initializers = A.Fake<Initializers>();
            var sigmoid = new ActivationFunction(node => node);
            var randomNormal = new InitializationFunction(shape => A.Fake<ExecutableModelNode>());

            A.CallTo(
                () => initializers.RandomNormal(A<float>.Ignored, A<float>.Ignored, A<float?>.Ignored)
            ).Returns(randomNormal);

            A.CallTo(() => backend.Activations).Returns(activations);
            A.CallTo(() => activations.Sigmoid()).Returns(sigmoid);

            return backend;
        }

        [Fact]
        public void WhenCompiledReturnsExecutableNode()
        {
            var backend = CreateBackend();
            var layer = new Dense("Test", 10, backend.Activations.Sigmoid(), new Input(new long[] { -1, 20 }));

            var executableNode = layer.Compile(backend);

            executableNode.Should().NotBeNull();
        }

        [Fact]
        public void WhenCompiledConfiguresCorrectOutputShape()
        {
            var backend = CreateBackend();
            var layer = new Dense("Test", 10, backend.Activations.Sigmoid(), new Input(new long[] { -1, 20 }));

            var executableNode = layer.Compile(backend);

            layer.Shape.Should().BeEquivalentTo(new long[] { -1, 10 });
        }
    }
}
