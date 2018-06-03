using FluentAssertions;
using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;

namespace Neuromatic.TensorFlow.Tests
{
    public class TensorFlowBackendFunctionSpecs
    {
        [Fact]
        public void ReturnsOutputValues()
        {
            using (var backend = new TensorFlowModelBackend())
            {
                var a = backend.Placeholder("A", new long[] { 1 });
                var b = backend.Placeholder("B", new long[] { 1 });

                var add = backend.Add(a, b, "Add");

                var inputMapping = new Dictionary<Core.ExecutableModelNode, object>
                {
                    { a, 2.0f },
                    { b, 3.0f }
                };

                var outputs = new[] { add };

                var func = backend.Function(new[] { a, b }, outputs, Enumerable.Empty<ExecutableModelNode>());

                var outputValue = func.Execute(new object[] { 2.0f, 3.0f });

                outputValue.Should().BeEquivalentTo(5.0f);
            }
        }


        [Fact]
        public void FailsWhenExecutedWithInvalidType()
        {
            using (var backend = new TensorFlowModelBackend())
            {
                var a = backend.Placeholder("A", new long[] { 1 });
                var b = backend.Placeholder("B", new long[] { 1 });

                var add = backend.Add(a, b, "Add");

                var outputs = new[] { add };
                var inputValues = new object[] { 2.0f, "world" };

                var func = backend.Function(new[] { a, b }, outputs, Enumerable.Empty<ExecutableModelNode>());

                func.Invoking(x => x.Execute(inputValues)).Should().Throw<ArgumentException>();
            }
        }

        [Fact]
        public void AcceptsFloatArrays()
        {
            using (var backend = new TensorFlowModelBackend())
            {
                var a = backend.Placeholder("A", new long[] { 1 });
                var b = backend.Placeholder("B", new long[] { 1 });

                var add = backend.Add(a, b, "Add");
                var outputs = new[] { add };

                var func = backend.Function(new[] { a, b }, outputs, Enumerable.Empty<ExecutableModelNode>());

                float[][] inputValues = new[]
                {
                    new[] { 2.0f, 4.0f },
                    new[] { 3.0f, 6.0f }
                };

                float[] outputValue = (float[])func.Execute(inputValues).ElementAt(0);

                outputValue[0].Should().Be(5.0f);
                outputValue[1].Should().Be(10.0f);
            }
        }
    }
}
