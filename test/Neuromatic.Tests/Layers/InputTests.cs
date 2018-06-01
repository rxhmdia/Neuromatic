using FakeItEasy;
using FluentAssertions;
using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace Neuromatic.Tests.Layers
{
    public class InputTests
    {
        [Fact]
        public void WhenCompiledReturnsPlaceholders()
        {
            var backend = A.Fake<ModelBackend>();

            var input = new Input(new long[] {  -1, 20});
            var executableNode = input.Compile(backend);

            executableNode.Should().NotBeNull();
        }

        [Fact]
        public void WhenCompiledCreatesPlaceholder()
        {
            var backend = A.Fake<ModelBackend>();

            var input = new Input(new long[] { -1, 20 });
            var executableNode = input.Compile(backend);

            A.CallTo(() => backend.Placeholder(A<string>.Ignored, A<long[]>.Ignored)).MustHaveHappened();
        }
    }
}
