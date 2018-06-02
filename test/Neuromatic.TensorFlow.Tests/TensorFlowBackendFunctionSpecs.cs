using FluentAssertions;
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

                var outputs = new [] { add };

                var func = backend.Function(inputMapping, outputs);

                var outputValue = func.Execute();

                outputValue.Should().BeEquivalentTo(5.0f);
            }
        }

        [Fact]
        public void RespectsOverrides()
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

                var overrides = new Dictionary<Core.ExecutableModelNode, object>
                {
                    {b, 20.0f }
                };

                var outputs = new[] { add };

                var func = backend.Function(inputMapping, outputs);

                var outputValue = func.Execute(overrides);

                outputValue.Should().BeEquivalentTo(22.0f);
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

                var inputMapping = new Dictionary<Core.ExecutableModelNode, object>
                {
                    { a, 2.0f },
                    { b, "world" }
                };

                var outputs = new[] { add };

                var func = backend.Function(inputMapping, outputs);

                func.Invoking(x => x.Execute()).Should().Throw<ArgumentException>();
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

                var inputMapping = new Dictionary<Core.ExecutableModelNode, object>
                {
                    { a, new[] { 2.0f, 4.0f } },
                    { b, new[] { 3.0f, 6.0f } }
                };

                var outputs = new[] { add };

                var func = backend.Function(inputMapping, outputs);

                float[] outputValue = (float[])func.Execute().ElementAt(0);

                outputValue[0].Should().Be(5.0f);
                outputValue[1].Should().Be(10.0f);
            }
        }
    }
}
