using FakeItEasy;
using Neuromatic.Core;
using Neuromatic.Initializers;
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
            
            return backend;
        }

        [Fact]
        public void WhenCompiledReturnsExecutableNode()
        {
            var backend = CreateBackend();
            var layer = new Dense(10, StandardActivations.Sigmoid(), new Input(new long[] { -1, 20 }), "Test");

            var executableNode = layer.Compile(backend);

            executableNode.Should().NotBeNull();
        }

        [Fact]
        public void WhenCompiledConfiguresCorrectOutputShape()
        {
            var backend = CreateBackend();
            var layer = new Dense(10, StandardActivations.Sigmoid(), new Input(new long[] { -1, 20 }), "Test");

            var executableNode = layer.Compile(backend);

            layer.Shape.Should().BeEquivalentTo(new long[] { -1, 10 });
        }
    }
}
